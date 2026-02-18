"""
Phase 3 — Magnet: "Campo Magnético Terrestre"

Effect: Applies a constant attraction force to the puck toward the collector's mallet.
        on_tick is called every frame by the manager.
        Force is read via EffectStack.get_magnet_force().
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING

from shared.powerups.registry import PowerUpDefinition
from shared.powerups.config import POWERUP_CONFIGS

if TYPE_CHECKING:
    from shared.powerups.registry import PowerUpRegistry

_CFG = POWERUP_CONFIGS["magnet"]
_BASE_FORCE = _CFG["force"]
_MIN_DIST   = 10  # px — prevents infinite force at point-blank


def _on_collect(collector_idx: int, players: list, puck, state) -> None:
    """Magnet is driven by on_tick; nothing to set up here."""


def _on_expire(target_idx: int, players: list, puck, state) -> None:
    """Force simply stops being applied when effect leaves the stack."""


def _on_tick(dt: float, target_idx: int, players: list, puck, state) -> None:
    """Apply puck attraction toward the target player's mallet every frame."""
    if target_idx >= len(players):
        return
    player = players[target_idx]
    dx = player.position[0] - puck.position[0]
    dy = player.position[1] - puck.position[1]
    dist = math.hypot(dx, dy)
    if dist <= _MIN_DIST:
        return
    # Force magnitude is taken from the EffectStack to support Duplication
    stack = getattr(state, "stacks", None)
    if stack is not None:
        force = stack[target_idx].get_magnet_force()
    else:
        force = _BASE_FORCE
    puck.velocity[0] += (dx / dist) * force * dt * 60  # normalize to 60 fps
    puck.velocity[1] += (dy / dist) * force * dt * 60


def build() -> PowerUpDefinition:
    defn = PowerUpDefinition(
        id           = "magnet",
        name         = "Campo Magnético Terrestre",
        description  = "Atrae el puck (radio +8px)",
        help_text    = (
            "El campo magnético terrestre crea un aura de atracción alrededor "
            "de tu mazo. El anillo animado indica el área de influencia activa."
        ),
        strategy_tip = (
            "Combinado con Velocidad Solar, permite controlar el puck y golpear "
            "con máxima potencia. En defensa, aleja el puck de tu portería pasivamente."
        ),
        duration      = _CFG["duration"],
        phase         = _CFG["phase"],
        color         = (170, 50, 220),
        icon          = "magnet",
        particle_color= (170, 50, 220),
        sound_key     = "powerup_magnet",
        # magnet_force → sum-accumulated by EffectStack.get_total("magnet_force")
        numeric_contributions = {"magnet_force": _BASE_FORCE},
        on_collect    = _on_collect,
        on_expire     = _on_expire,
        on_tick       = _on_tick,
        affects_self  = True,
        can_stack     = True,
    )
    return defn


def register(registry: "PowerUpRegistry") -> None:
    registry.register(build())
