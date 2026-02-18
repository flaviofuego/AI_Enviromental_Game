"""
Phase 8 — Invisibility: "Capa de Invisibilidad"

Effect: The collector's mallet is rendered with very low alpha (20) on the
        opponent's screen.  In RL mode, the human player's coordinates are
        replaced by Gaussian noise in the observation vector.

        State flags used:
            state.player_invisible[collector_idx] = True / False
        The renderer reads this flag to set sprite alpha.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from shared.powerups.registry import PowerUpDefinition
from shared.powerups.config import POWERUP_CONFIGS

if TYPE_CHECKING:
    from shared.powerups.registry import PowerUpRegistry

_CFG = POWERUP_CONFIGS["invisibility"]


def _on_collect(collector_idx: int, players: list, puck, state) -> None:
    """Mark the collector as invisible in state flags."""
    if not hasattr(state, "player_invisible"):
        state.player_invisible = [False, False]
    state.player_invisible[collector_idx] = True


def _on_expire(target_idx: int, players: list, puck, state) -> None:
    """Restore visibility."""
    player_invisible = getattr(state, "player_invisible", None)
    if player_invisible is not None:
        player_invisible[target_idx] = False


def build() -> PowerUpDefinition:
    return PowerUpDefinition(
        id           = "invisibility",
        name         = "Capa de Invisibilidad",
        description  = "Tu mazo es invisible al rival",
        help_text    = (
            "Como la capa de ozono que oculta la Tierra de la radiación, "
            "este powerup te hace casi imperceptible para el oponente durante "
            "3 segundos. Úsalo para ataques sorpresa."
        ),
        strategy_tip  = "Efectivo para movimientos de ataque fintas. El oponente no puede predecir tu posición exacta.",
        duration      = _CFG["duration"],
        phase         = _CFG["phase"],
        color         = (220, 220, 255),
        icon          = "invisibility",
        particle_color= (220, 220, 255),
        sound_key     = "powerup_invisible",
        on_collect    = _on_collect,
        on_expire     = _on_expire,
        on_tick       = None,
        affects_self  = True,
        can_stack     = True,  # second instance → super-invisibility (alpha=10)
    )


def register(registry: "PowerUpRegistry") -> None:
    registry.register(build())
