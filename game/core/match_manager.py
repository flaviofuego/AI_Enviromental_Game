"""
Match manager: orchestrates game modes, win conditions, and match flow.
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class GameMode(Enum):
    """Available game modes."""
    PLAYER_VS_AI = auto()
    PLAYER_VS_PLAYER = auto()


@dataclass
class MatchConfig:
    """Configuration for a single match."""
    mode: GameMode = GameMode.PLAYER_VS_AI
    score_limit: int = 7
    time_limit_seconds: Optional[float] = None
    overtime_on_tie: bool = True
    powerups_enabled: bool = False
    level_id: int = 1
    ai_difficulty: str = "standard"  # Maps to level config


class MatchManager:
    """Manages match lifecycle: start, update, win condition checking."""

    def __init__(self, config: MatchConfig = None):
        self.config = config or MatchConfig()
        self.is_active = False

    @property
    def mode(self) -> GameMode:
        return self.config.mode

    @property
    def score_limit(self) -> int:
        return self.config.score_limit

    @property
    def time_limit(self) -> Optional[float]:
        return self.config.time_limit_seconds

    @property
    def powerups_enabled(self) -> bool:
        return self.config.powerups_enabled

    def start_match(self):
        self.is_active = True

    def end_match(self):
        self.is_active = False
