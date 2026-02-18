"""
PowerUpDefinition and PowerUpRegistry — pure data contracts.

Every powerup is a *data record* with injected callable components
(on_collect, on_expire, on_tick).  The manager knows nothing about
specific powerup logic; it only dispatches to these callables.

Design principles:
- Composition over inheritance: behavior injected as callables, not subclasses.
- Single Responsibility: this module only defines the contract + catalogue.
- KISS: dataclass + dict; no metaclass magic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Type aliases for readability
# ---------------------------------------------------------------------------

#: Signature for on_collect / on_expire.
#:   collector_idx  – index of the player who collected the powerup (0 or 1)
#:   players        – list of mallet entities
#:   puck           – the puck entity
#:   state          – game state object (used to read/write score, flags, etc.)
EffectCallback = Callable[[int, list, object, object], None]

#: Signature for on_tick (called every frame while the effect is active).
#:   dt             – seconds since last frame
#:   target_idx     – index of the player affected
#:   players, puck, state – same as above
TickCallback = Callable[[float, int, list, object, object], None]


# ---------------------------------------------------------------------------
# PowerUpDefinition
# ---------------------------------------------------------------------------

@dataclass
class PowerUpDefinition:
    """
    Immutable descriptor for a single powerup type.

    All game-logic callables (on_collect, on_expire, on_tick) are injected
    at registration time by the definition module (e.g. phase1_speed.py).
    The PowerUpManager treats every definition identically: it calls the
    right hook at the right time.
    """

    # --- Identity ---------------------------------------------------------
    id: str
    """Unique snake_case identifier, e.g. 'speed_boost'."""

    name: str
    """Thematic display name, e.g. 'Viento Solar'."""

    description: str
    """Short HUD label, e.g. '+30% fuerza de golpe'."""

    help_text: str
    """Multi-sentence explanation shown in the help panel."""

    strategy_tip: str
    """One-line strategic advice shown in the help panel."""

    # --- Timing & Phase ---------------------------------------------------
    duration: float
    """Active effect duration in seconds."""

    phase: int
    """Phase number (1–8) at which this powerup unlocks."""

    # --- Visual identity --------------------------------------------------
    color: tuple
    """RGB tuple for the field sphere, HUD pill, and particles."""

    icon: str
    """Emoji or ASCII character used in HUD and help panel."""

    particle_color: tuple = field(default_factory=lambda: (255, 255, 255))
    """Particle burst color on collection (defaults to white)."""

    sound_key: str = ""
    """AudioManager key, e.g. 'powerup_speed'.  Empty = no sound."""

    # --- Numeric contributions (read by EffectStack generically) ----------
    numeric_contributions: dict = field(default_factory=dict)
    """
    Maps a string key to a float value that EffectStack aggregates.

    Multiplicative effects use the key without suffix:
        {"speed_mult": 1.3}   → product-accumulates in get_multiplier("speed_mult")
        {"size_mult":  1.3}   → product-accumulates in get_multiplier("size_mult")

    Additive effects append "_add":
        {"magnet_force": 0.3} → sum-accumulates in get_total("magnet_force")

    Adding a new numeric powerup only requires filling this dict —
    EffectStack needs no changes.
    """

    # --- Effect callables (SRP: logic lives in definition modules) --------
    on_collect: EffectCallback = field(default=lambda ci, pl, pk, st: None)
    """Called once when a player collects this powerup."""

    on_expire: EffectCallback = field(default=lambda ci, pl, pk, st: None)
    """Called once when the effect timer hits zero."""

    on_tick: Optional[TickCallback] = None
    """Called every frame while the effect is active (None = no per-frame logic)."""

    # --- Behaviour flags --------------------------------------------------
    affects_self: bool = True
    """True → target is the collector; False → target is the opponent."""

    can_stack: bool = True
    """Whether multiple instances of this powerup can be active simultaneously."""


# ---------------------------------------------------------------------------
# PowerUpRegistry
# ---------------------------------------------------------------------------

class PowerUpRegistry:
    """
    Central catalogue of all registered PowerUpDefinition instances.

    Registration is intentionally additive: each phase module calls
    ``registry.register(definition)`` at import time.  The manager only
    needs to call ``registry.get(id)`` or ``registry.get_all()``.

    Example:
        registry = PowerUpRegistry()
        from shared.powerups.definitions import phase1_speed
        phase1_speed.register(registry)
        defn = registry.get("speed_boost")
    """

    def __init__(self) -> None:
        self._definitions: Dict[str, PowerUpDefinition] = {}

    # --- Mutation ---------------------------------------------------------

    def register(self, definition: PowerUpDefinition) -> None:
        """Add a PowerUpDefinition.  Raises if id is already taken."""
        if definition.id in self._definitions:
            raise ValueError(
                f"PowerUpDefinition '{definition.id}' is already registered. "
                "Use a unique id per powerup."
            )
        self._definitions[definition.id] = definition

    def unregister(self, powerup_id: str) -> None:
        """Remove a definition by id (useful for testing)."""
        self._definitions.pop(powerup_id, None)

    # --- Queries ----------------------------------------------------------

    def get(self, powerup_id: str) -> PowerUpDefinition:
        """Return the definition for *powerup_id*, or raise KeyError."""
        try:
            return self._definitions[powerup_id]
        except KeyError:
            raise KeyError(
                f"No PowerUpDefinition registered with id='{powerup_id}'. "
                f"Available: {list(self._definitions.keys())}"
            )

    def get_all(self) -> List[PowerUpDefinition]:
        """Return all definitions in registration order."""
        return list(self._definitions.values())

    def get_by_phase(self, phase: int) -> List[PowerUpDefinition]:
        """Return all definitions belonging to *phase*."""
        return [d for d in self._definitions.values() if d.phase == phase]

    def get_all_by_phase(self) -> List[PowerUpDefinition]:
        """Return all definitions sorted by phase then id."""
        return sorted(self._definitions.values(), key=lambda d: (d.phase, d.id))

    def ids(self) -> List[str]:
        """Return all registered ids."""
        return list(self._definitions.keys())

    def __len__(self) -> int:
        return len(self._definitions)

    def __repr__(self) -> str:  # pragma: no cover
        ids = list(self._definitions.keys())
        return f"PowerUpRegistry({ids})"
