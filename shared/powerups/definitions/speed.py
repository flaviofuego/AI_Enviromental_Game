"""
Phase 1 — Speed Boost: "Viento Solar"

Effect: +30 % hit-force multiplier on the collector's puck impacts.
Composable: multiple instances multiply (1.3 × 1.3 = 1.69).
Expiry: removes this effect's contribution from EffectStack — other
    strike effects continue to apply.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from shared.powerups.registry import PowerUpDefinition
from shared.powerups.config import POWERUP_CONFIGS

if TYPE_CHECKING:
    from shared.powerups.registry import PowerUpRegistry

_CFG = POWERUP_CONFIGS["speed_boost"]


# ---------------------------------------------------------------------------
# Effect callbacks
# ---------------------------------------------------------------------------

def _on_collect(collector_idx: int, players: list, puck, state) -> None:
    """Strike boost is applied via EffectStack.get_strike_multiplier()."""
    # The manager reads get_strike_multiplier() every frame and sets
    # player.strike_multiplier from it.  Nothing to do here beyond the
    # EffectStack entry the manager already created before calling this hook.


def _on_expire(target_idx: int, players: list, puck, state) -> None:
    """Also handled by EffectStack recompute — no explicit reset needed."""


# ---------------------------------------------------------------------------
# Definition
# ---------------------------------------------------------------------------

def build() -> PowerUpDefinition:
    defn = PowerUpDefinition(
        id          = "speed_boost",
        name        = "Viento Solar",
        description = "+30% fuerza de golpe",
        help_text   = (
            "El viento solar carga tu mazo de energía cinética, "
            "incrementando la potencia de cada golpe. "
            "Cronometra su uso para los momentos de ataque."
        ),
        strategy_tip = (
            "Recoger justo antes de un ataque para maximizar la fuerza "
            "del golpe. Combinado con el Imán, da control total del puck."
        ),
        duration      = _CFG["duration"],
        phase         = _CFG["phase"],
        color         = (0, 200, 255),
        icon          = "speed_boost",
        particle_color= (0, 200, 255),
        sound_key     = "powerup_speed",
        # strike_mult → product-accumulated by EffectStack.get_multiplier("strike_mult")
        numeric_contributions = {"strike_mult": _CFG["multiplier"]},
        on_collect    = _on_collect,
        on_expire     = _on_expire,
        on_tick       = None,
        affects_self  = True,
        can_stack     = True,
    )
    return defn


def register(registry: "PowerUpRegistry") -> None:
    registry.register(build())
