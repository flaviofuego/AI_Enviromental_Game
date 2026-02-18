"""
Phase 7 — Paralyze: "Tormenta Eléctrica"

Effect: Completely freezes the opponent's mallet for the duration.
        The mallet entity checks EffectStack.is_paralyzed() every frame.
        In RL mode, the manager replaces the action with "stay" (action 4).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from shared.powerups.registry import PowerUpDefinition
from shared.powerups.config import POWERUP_CONFIGS

if TYPE_CHECKING:
    from shared.powerups.registry import PowerUpRegistry

_CFG = POWERUP_CONFIGS["paralyze"]


def _on_collect(collector_idx: int, players: list, puck, state) -> None:
    """Paralysis is checked via EffectStack.is_paralyzed() — no direct attribute mutation."""


def _on_expire(target_idx: int, players: list, puck, state) -> None:
    """Movement automatically resumes once the effect leaves the stack."""


def build() -> PowerUpDefinition:
    return PowerUpDefinition(
        id           = "paralyze",
        name         = "Tormenta Eléctrica",
        description  = "Oponente paralizado 3s",
        help_text    = (
            "La tormenta eléctrica descarga sobre el mazo del oponente, "
            "dejándolo sin control. Aprovecha estos 3 segundos para anotar "
            "goles sin resistencia."
        ),
        strategy_tip  = "Activar y atacar directamente. Tienes 3 segundos de ventaja táctica absoluta.",
        duration      = _CFG["duration"],
        phase         = _CFG["phase"],
        color         = (255, 240, 0),
        icon          = "paralyze",
        particle_color= (255, 240, 0),
        sound_key     = "powerup_paralyze",
        on_collect    = _on_collect,
        on_expire     = _on_expire,
        on_tick       = None,
        affects_self  = False,   # targets the opponent
        can_stack     = True,    # second instance extends duration
    )


def register(registry: "PowerUpRegistry") -> None:
    registry.register(build())
