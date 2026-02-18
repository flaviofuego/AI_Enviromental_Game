"""
Phase definitions package.

Each sub-module exposes a single ``register(registry)`` function, making
it trivial to enable or disable phases without touching the manager:

    from shared.powerups.definitions import phase1_speed
    phase1_speed.register(registry)

To activate all phases at once use ``register_all(registry)``.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.powerups.registry import PowerUpRegistry


def register_all(registry: "PowerUpRegistry") -> None:
    """Register every phase definition into *registry* in phase order."""
    from shared.powerups.definitions import (
        speed,
        shield,
        magnet,
        duplicate,
        slow,
        obstacle,
        paralyze,
        invisibility,
    )
    for mod in (
        speed,
        shield,
        magnet,
        duplicate,
        slow,
        obstacle,
        paralyze,
        invisibility,
    ):
        mod.register(registry)


def register_phases(registry: "PowerUpRegistry", phases: list[int]) -> None:
    """Register only the definition modules corresponding to *phases*."""
    phase_map = {
        1: "speed",
        2: "shield",
        3: "magnet",
        4: "duplicate",
        5: "slow",
        6: "obstacle",
        7: "paralyze",
        8: "invisibility",
    }
    import importlib
    for p in sorted(set(phases)):
        mod_name = phase_map.get(p)
        if mod_name is None:
            raise ValueError(f"Unknown phase number: {p}")
        mod = importlib.import_module(f"shared.powerups.definitions.{mod_name}")
        mod.register(registry)
