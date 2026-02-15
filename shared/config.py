"""
Centralized configuration for the Air Hockey game.
Replaces constants.py with dataclasses — no mutable globals, no pygame dependency at import time.
Dimensions are passed as parameters at runtime.
"""
import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Color constants (no pygame dependency)
# ---------------------------------------------------------------------------

class COLORS:
    """Named color constants used throughout the project."""
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    NEON_RED = (255, 60, 60)
    NEON_GREEN = (60, 255, 60)
    NEON_BLUE = (60, 60, 255)


# ---------------------------------------------------------------------------
# Game configuration
# ---------------------------------------------------------------------------

@dataclass
class GameConfig:
    """Runtime game configuration.  Create once and pass around."""
    width: int = 800
    height: int = 500
    fps: int = 120

    # Derived (computed in __post_init__)
    half_width: int = field(init=False)
    scale_factor: float = field(init=False)
    puck_radius: int = field(init=False)
    mallet_radius: int = field(init=False)

    def __post_init__(self):
        self.half_width = self.width // 2
        sf = min(self.width / 800, self.height / 500)
        self.scale_factor = sf
        self.puck_radius = max(8, int(15 * sf))
        self.mallet_radius = max(16, int(32 * sf))


# ---------------------------------------------------------------------------
# Physics configuration
# ---------------------------------------------------------------------------

@dataclass
class PhysicsConfig:
    """Physics parameters for the hockey simulation."""
    friction: float = 0.9999
    max_puck_speed: float = 12.0
    collision_elasticity: float = 0.8
    goal_width_ratio: float = 1 / 3
    ai_move_amount: int = 5  # Must match training env (base_env.py)


# ---------------------------------------------------------------------------
# Training reference dimensions (what the RL model learned with)
# ---------------------------------------------------------------------------
TRAINING_WIDTH = 800
TRAINING_HEIGHT = 500


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
IMPROVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "improved_models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
