"""
TrainingState — lightweight game-state proxy for training environments.

The PowerUpManager (shared/) requires a *state* object it can read/write to
communicate field geometry, obstacles, and per-player stacks.  In the full
game this is GameState (pygame-heavy).  Here we provide a plain dataclass
that satisfies the same interface without any pygame dependency.

Design: Single Responsibility — this file only holds state that the
PowerUpManager and its definition callbacks may read or mutate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from shared.powerups.effect_stack import EffectStack


@dataclass
class TrainingState:
    """
    Minimal game-state object compatible with PowerUpManager callbacks.

    Attributes
    ----------
    width, height       : field dimensions in pixels
    stacks              : set by PowerUpManager.update() to expose EffectStacks
    obstacles           : obstacle data placed by the Obstacle powerup (phase 6).
                          Each entry is a dict {"position": [x,y], "radius": r}.
    _shield_blocked     : internal flag toggled by shield callbacks to signal
                          that the next goal should be blocked.
    """

    width: float
    height: float

    # Injected by PowerUpManager each update() call
    stacks: List["EffectStack"] = field(default_factory=list)

    # Obstacle powerup state (phase 6)
    obstacles: List[dict] = field(default_factory=list)

    # Shield state — index → bool
    shield_active: List[bool] = field(default_factory=lambda: [False, False])

    # Score counters (mirrored from env for callbacks that need them)
    player_score: int = 0
    ai_score: int = 0

    def reset(self) -> None:
        """Clear mutable state at episode boundary."""
        self.obstacles.clear()
        self.shield_active = [False, False]
        self.player_score = 0
        self.ai_score = 0
        # stacks is re-injected by the manager each update, no need to clear
