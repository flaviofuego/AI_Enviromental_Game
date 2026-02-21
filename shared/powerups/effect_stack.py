"""
EffectStack — per-player accumulator of active powerup effects.

Design:
- Each player owns one EffectStack.
- Effects *multiply* instead of overwrite (composable).
- Getters (get_speed_multiplier, etc.) compute the combined value on-demand
  from all active effects — O(n) where n is small (< 10 in practice).
- stacked_multiplier supports the Duplication powerup (phase 4).

Principle: Single Responsibility.  This module only manages *what effects
are active* and *what their combined numeric impact is*.  It does NOT know
about pygame, rendering, or game rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.powerups.registry import PowerUpDefinition


# ---------------------------------------------------------------------------
# MIN/MAX caps — prevents degenerate gameplay
# ---------------------------------------------------------------------------

SPEED_MIN_CAP    = 0.40   # phase-5 slow_opponent: can't go below 40 % speed
SPEED_MAX_CAP    = 5.00   # reasonable upper limit to prevent broken combos
SIZE_MIN_CAP     = 0.30
SIZE_MAX_CAP     = 3.00
MAGNET_FORCE_CAP = 2.0    # phase-3 + duplication cap

# Generic cap table: any numeric key registered here is clamped automatically.
# Add entries here when creating numeric powerups that need safety limits;
# keys absent from the table are returned unclamped.
_MULTIPLIER_CAPS: dict = {
    "speed_mult": (SPEED_MIN_CAP, SPEED_MAX_CAP),
    "strike_mult": (1.0, SPEED_MAX_CAP),
    "size_mult":  (SIZE_MIN_CAP,  SIZE_MAX_CAP),
}
_TOTAL_CAPS: dict = {
    "magnet_force": (0.0, MAGNET_FORCE_CAP),
}


# ---------------------------------------------------------------------------
# ActiveEffect
# ---------------------------------------------------------------------------

@dataclass
class ActiveEffect:
    """
    A single in-flight powerup effect bound to a player's EffectStack.

    ``stacked_multiplier`` is mutated by the Duplication powerup (phase 4)
    to amplify numeric contributions without touching the base definition.
    """

    definition: "PowerUpDefinition"
    remaining: float
    """Seconds of effect time left."""

    collector_idx: int
    """Index (0 or 1) of the player who collected this powerup."""

    target_idx: int
    """Index (0 or 1) of the player who is affected."""

    stacked_multiplier: float = 1.0
    """Applied on top of the definition's intrinsic multiplier by Duplication."""

    # --- Convenience accessors -------------------------------------------

    @property
    def id(self) -> str:
        return self.definition.id

    @property
    def is_expired(self) -> bool:
        return self.remaining <= 0.0

    def tick(self, dt: float) -> None:
        self.remaining -= dt

    def __repr__(self) -> str:
        return (
            f"ActiveEffect(id={self.id!r}, remaining={self.remaining:.2f}s, "
            f"target={self.target_idx}, stacked_mult={self.stacked_multiplier})"
        )


# ---------------------------------------------------------------------------
# EffectStack
# ---------------------------------------------------------------------------

class EffectStack:
    """
    Ordered collection of ActiveEffects for a single player.

    All numeric getters return the *product* of every contributing effect,
    so powerups compose naturally (each adds a multiplicative layer).

    Thread-safety is not a concern here (single-threaded main loop).
    """

    def __init__(self) -> None:
        self._effects: List[ActiveEffect] = []

    # --- Mutation ---------------------------------------------------------

    def add(self, effect: ActiveEffect) -> None:
        """Add a new active effect to this player's stack."""
        self._effects.append(effect)

    def remove(self, effect: ActiveEffect) -> None:
        """Remove an effect (called after expiry + on_expire hook)."""
        try:
            self._effects.remove(effect)
        except ValueError:
            pass

    def clear(self) -> None:
        """Remove all effects (e.g. on match reset)."""
        self._effects.clear()

    # --- Tick (called every frame by PowerUpManager) ----------------------

    def tick(self, dt: float) -> List[ActiveEffect]:
        """
        Advance all timers by *dt* seconds.

        Returns the list of effects that *just expired* this frame so the
        manager can fire on_expire callbacks.
        """
        just_expired: List[ActiveEffect] = []
        for eff in self._effects:
            eff.tick(dt)
            if eff.is_expired:
                just_expired.append(eff)
        return just_expired

    # --- Numeric queries (composable multipliers) -------------------------

    def get_multiplier(self, key: str, default: float = 1.0) -> float:
        """
        Generic product-accumulator for any multiplicative numeric key.

        Reads ``definition.numeric_contributions[key]`` from every active
        effect, multiplies them together (each scaled by ``stacked_multiplier``),
        and clamps the result using ``_MULTIPLIER_CAPS``.

        Adding a new multiplicative powerup requires only filling
        ``numeric_contributions`` in its definition — no changes here.
        """
        acc = default
        for eff in self._effects:
            val = eff.definition.numeric_contributions.get(key)
            if val is not None:
                acc *= val * eff.stacked_multiplier
        lo, hi = _MULTIPLIER_CAPS.get(key, (float("-inf"), float("inf")))
        return max(lo, min(hi, acc))

    def get_total(self, key: str) -> float:
        """
        Generic sum-accumulator for any additive numeric key.

        Reads ``definition.numeric_contributions[key]`` from every active
        effect, sums them (each scaled by ``stacked_multiplier``), and clamps
        the result using ``_TOTAL_CAPS``.
        """
        total = 0.0
        for eff in self._effects:
            val = eff.definition.numeric_contributions.get(key)
            if val is not None:
                total += val * eff.stacked_multiplier
        lo, hi = _TOTAL_CAPS.get(key, (0.0, float("inf")))
        return max(lo, min(hi, total))

    # --- Named wrappers (backward-compatible convenience) -----------------
    # These call the generic methods above.  External code (manager, engine)
    # can use either form; adding new powerup types never requires touching
    # EffectStack again.

    def get_speed_multiplier(self) -> float:
        """Product of all speed-affecting multipliers (clamped)."""
        return self.get_multiplier("speed_mult")

    def get_size_multiplier(self) -> float:
        """Product of all size-affecting multipliers (clamped)."""
        return self.get_multiplier("size_mult")

    def get_strike_multiplier(self) -> float:
        """Product of all hit-force multipliers applied to puck impacts."""
        return self.get_multiplier("strike_mult")

    def get_magnet_force(self) -> float:
        """Sum of all magnet attraction forces (clamped)."""
        return self.get_total("magnet_force")

    # --- Boolean state queries -------------------------------------------

    def has_shield(self) -> bool:
        """True if at least one shield effect is active."""
        return any(eff.id == "shield" for eff in self._effects)

    def has_magnet(self) -> bool:
        """True if at least one magnet effect is active."""
        return any(eff.id == "magnet" for eff in self._effects)

    def is_paralyzed(self) -> bool:
        """True if the player cannot move (paralyze powerup)."""
        return any(eff.id == "paralyze" for eff in self._effects)

    def is_invisible(self) -> bool:
        """True if this player's mallet should be hidden from opponent."""
        return any(eff.id == "invisibility" for eff in self._effects)

    def has_obstacles(self) -> bool:
        """True if this player spawned obstacles that are still active."""
        return any(eff.id == "obstacle" for eff in self._effects)

    # --- Duplication support ---------------------------------------------

    def apply_stacked_multiplier(self, factor: float) -> None:
        """
        Multiply ``stacked_multiplier`` on all current effects by *factor*.
        Called by the Duplication powerup (phase 4) on_collect.
        """
        for eff in self._effects:
            eff.stacked_multiplier *= factor

    # --- Convenience / introspection ------------------------------------

    def has(self, powerup_id: str) -> bool:
        """Return True if at least one active effect with *powerup_id* exists."""
        return any(e.id == powerup_id for e in self._effects)

    def get_effects_by_id(self, powerup_id: str) -> List[ActiveEffect]:
        """Return all active effects with the given id."""
        return [e for e in self._effects if e.id == powerup_id]

    def get_shortest_remaining(self, powerup_id: str) -> Optional[float]:
        """Return the minimum remaining time for effects of *powerup_id*, or None."""
        candidates = [e.remaining for e in self._effects if e.id == powerup_id]
        return min(candidates) if candidates else None

    @property
    def effects(self) -> List[ActiveEffect]:
        """Read-only view of current effects (copy prevents external mutation)."""
        return list(self._effects)

    def __len__(self) -> int:
        return len(self._effects)

    def __bool__(self) -> bool:
        return bool(self._effects)

    def __repr__(self) -> str:
        return f"EffectStack([{', '.join(e.id for e in self._effects)}])"
