"""
Phase 6 — Obstacle: "Iceberg Flotante"

Effect: Spawns 2 circular obstacles in the central field zone.
        The puck bounces off them (fully elastic); mallets pass through.
        Obstacles are stored on *state* so the renderer and engine can
        access them without importing pygame in this module.
"""
from __future__ import annotations
import random
from typing import TYPE_CHECKING

from shared.powerups.registry import PowerUpDefinition
from shared.powerups.config import POWERUP_CONFIGS

if TYPE_CHECKING:
    from shared.powerups.registry import PowerUpRegistry

_CFG   = POWERUP_CONFIGS["obstacle"]
_COUNT = _CFG["count"]
_RADIUS = _CFG["radius"]

# Maximum simultaneous obstacles on field (across all collectors)
MAX_FIELD_OBSTACLES = 6


def _spawn_obstacles(state, collector_idx: int) -> list:
    """Create obstacle data dicts and attach to state.obstacles."""
    W = getattr(state, "width",  800)
    H = getattr(state, "height", 500)

    if not hasattr(state, "obstacles"):
        state.obstacles = []

    new_obstacles = []
    for _ in range(_COUNT):
        if len(state.obstacles) >= MAX_FIELD_OBSTACLES:
            break
        obs = {
            "x":              random.uniform(W * 0.30, W * 0.70),
            "y":              random.uniform(H * 0.20, H * 0.80),
            "radius":         _RADIUS,
            "collector_idx":  collector_idx,
            "alpha":          255,
            "fading":         False,
            "still_timer":    0.0,  # used by anti-stuck mechanic
        }
        state.obstacles.append(obs)
        new_obstacles.append(obs)
    return new_obstacles


def _on_collect(collector_idx: int, players: list, puck, state) -> None:
    _spawn_obstacles(state, collector_idx)


def _on_expire(target_idx: int, players: list, puck, state) -> None:
    """Mark obstacles belonging to this collector as fading."""
    obstacles = getattr(state, "obstacles", [])
    for obs in obstacles:
        if obs.get("collector_idx") == target_idx:
            obs["fading"] = True


def build() -> PowerUpDefinition:
    return PowerUpDefinition(
        id           = "obstacle",
        name         = "Iceberg Flotante",
        description  = "Obstáculos temporales en campo",
        help_text    = (
            "El iceberg flotante coloca bloques de hielo en el campo que desvían "
            "el puck. Como los icebergs que obstruyen las rutas marítimas, estos "
            "obstáculos fuerzan nuevas estrategias de juego."
        ),
        strategy_tip = (
            "Activar cuando el oponente tiene control del campo. "
            "Los obstáculos desvían el puck de trayectorias predecibles."
        ),
        duration      = _CFG["duration"],
        phase         = _CFG["phase"],
        color         = (180, 230, 255),
        icon          = "obstacle",
        particle_color= (180, 230, 255),
        sound_key     = "powerup_obstacle",
        on_collect    = _on_collect,
        on_expire     = _on_expire,
        on_tick       = None,
        affects_self  = True,   # collector decides where obstacles appear
        can_stack     = True,
    )


def register(registry: "PowerUpRegistry") -> None:
    registry.register(build())
