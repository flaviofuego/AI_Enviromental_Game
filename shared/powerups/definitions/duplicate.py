"""
Phase 4 — Duplication: "Efecto Invernadero"

Effect: Doubles (stacked_multiplier × 2.0) the numeric multipliers of ALL
        currently active effects in the collector's EffectStack.
        Future effects collected during Duplication's lifetime are also doubled.

Implementation note:
    The manager passes the EffectStack reference as part of *state* so
    on_collect can mutate stacked_multiplier on existing effects.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from shared.powerups.registry import PowerUpDefinition
from shared.powerups.config import POWERUP_CONFIGS

if TYPE_CHECKING:
    from shared.powerups.registry import PowerUpRegistry

_CFG = POWERUP_CONFIGS["duplication"]


def _on_collect(collector_idx: int, players: list, puck, state) -> None:
    """Double all existing numeric multipliers in the collector's stack."""
    stacks = getattr(state, "stacks", None)
    if stacks is None:
        return
    stack = stacks[collector_idx]
    stack.apply_stacked_multiplier(_CFG["stack_mult"])


def _on_expire(target_idx: int, players: list, puck, state) -> None:
    """
    Revert the duplication factor (divide by stack_mult).

    Note: stacked_multiplier on each effect is persistent data; we divide
    it back to undo the amplification applied on collect.
    """
    stacks = getattr(state, "stacks", None)
    if stacks is None:
        return
    stack = stacks[target_idx]
    revert_factor = 1.0 / _CFG["stack_mult"]
    stack.apply_stacked_multiplier(revert_factor)


def build() -> PowerUpDefinition:
    return PowerUpDefinition(
        id           = "duplication",
        name         = "Efecto Invernadero",
        description  = "Duplica efectos activos",
        help_text    = (
            "El efecto invernadero amplifica todos tus poderes activos. "
            "Como los gases de efecto invernadero que amplifican el calor, "
            "este powerup duplica el impacto de cada beneficio que ya posees."
        ),
        strategy_tip = (
            "Recoger después de obtener Velocidad Solar o Imán para maximizar "
            "el combo. Es el powerup más valioso del campo."
        ),
        duration      = _CFG["duration"],
        phase         = _CFG["phase"],
        color         = (255, 210, 0),
        icon          = "duplication",
        particle_color= (255, 210, 0),
        sound_key     = "powerup_duplicate",
        on_collect    = _on_collect,
        on_expire     = _on_expire,
        on_tick       = None,
        affects_self  = True,
        can_stack     = True,
    )


def register(registry: "PowerUpRegistry") -> None:
    registry.register(build())
