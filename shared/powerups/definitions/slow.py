"""
Phase 5 — Slow Opponent: "Niebla Contaminante"

Effect: Reduces the opponent's speed_multiplier by 25 % (factor 0.75).
        Multiple instances accumulate: 0.75 × 0.75 = 0.5625.
        The EffectStack enforces a minimum cap of 0.40.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from shared.powerups.registry import PowerUpDefinition
from shared.powerups.config import POWERUP_CONFIGS

if TYPE_CHECKING:
    from shared.powerups.registry import PowerUpRegistry

_CFG = POWERUP_CONFIGS["slow_opponent"]


def _on_collect(collector_idx: int, players: list, puck, state) -> None:
    """Speed reduction is applied frame-by-frame via EffectStack.get_speed_multiplier()."""


def _on_expire(target_idx: int, players: list, puck, state) -> None:
    """Manager will recompute speed from remaining stack entries automatically."""


def build() -> PowerUpDefinition:
    defn = PowerUpDefinition(
        id           = "slow_opponent",
        name         = "Niebla Contaminante",
        description  = "-25% velocidad rival",
        help_text    = (
            "La niebla contaminante envuelve al oponente, ralentizando su mazo. "
            "Como la polución que frena el progreso, este powerup reduce la "
            "movilidad rival durante 7 segundos."
        ),
        strategy_tip = (
            "Ideal para fases de ataque sostenido. Combinado con Velocidad Solar "
            "propia, la diferencia de velocidad es máxima."
        ),
        duration      = _CFG["duration"],
        phase         = _CFG["phase"],
        color         = (130, 115, 100),
        icon          = "slow_opponent",
        particle_color= (130, 115, 100),
        sound_key     = "powerup_slow",
        # speed_mult < 1 → slows down; same key as speed_boost, they compose
        numeric_contributions = {"speed_mult": _CFG["multiplier"]},
        on_collect    = _on_collect,
        on_expire     = _on_expire,
        on_tick       = None,
        affects_self  = False,   # targets the opponent
        can_stack     = True,
    )
    return defn


def register(registry: "PowerUpRegistry") -> None:
    registry.register(build())
