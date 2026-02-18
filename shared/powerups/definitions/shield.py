"""
Phase 2 — Shield: "Barrera de Ozono"

Effect: Activates a shield on the collector's goal that intercepts the puck.
        The engine consults EffectStack.has_shield() in _check_goal().
Expiry: Shield disappears; goal detection resumes normally.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from shared.powerups.registry import PowerUpDefinition
from shared.powerups.config import POWERUP_CONFIGS

if TYPE_CHECKING:
    from shared.powerups.registry import PowerUpRegistry

_CFG = POWERUP_CONFIGS["shield"]


def _on_collect(collector_idx: int, players: list, puck, state) -> None:
    """Shield state is tracked via EffectStack.has_shield()."""


def _on_expire(target_idx: int, players: list, puck, state) -> None:
    """Shield state resolves automatically when effect is removed from stack."""


def build() -> PowerUpDefinition:
    return PowerUpDefinition(
        id           = "shield",
        name         = "Barrera de Ozono",
        description  = "Portería protegida",
        help_text    = (
            "La capa de ozono forma una barrera temporal que protege tu portería. "
            "El escudo rebota cualquier puck que intente atravesarla durante su duración."
        ),
        strategy_tip = (
            "Activarlo cuando el oponente está en posición de ataque directo. "
            "No protege esquinas extremas."
        ),
        duration      = _CFG["duration"],
        phase         = _CFG["phase"],
        color         = (255, 130, 60),
        icon          = "shield",
        particle_color= (255, 130, 60),
        sound_key     = "powerup_shield",
        on_collect    = _on_collect,
        on_expire     = _on_expire,
        on_tick       = None,
        affects_self  = True,
        can_stack     = False,  # second shield resets the timer, doesn't stack
    )


def register(registry: "PowerUpRegistry") -> None:
    registry.register(build())
