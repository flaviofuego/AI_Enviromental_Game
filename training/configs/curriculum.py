"""Curriculum learning configuration for progressive difficulty training.

Integrates with ``training.envs.opponents.OpponentParams`` to provide
fine-grained control over opponent behavior at each difficulty tier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DifficultyLevel:
    """A single difficulty tier with explicit opponent parameters."""
    level: int
    opponent_skill: float
    prediction_ability: float
    reaction_speed: float
    accuracy: float
    aggression: float

    # Advancement criteria
    min_win_rate_to_advance: float = 0.6
    min_avg_goals_scored: float = 0.0    # NEW: avg goals per episode
    max_avg_goals_conceded: float = 10.0  # NEW: max avg goals conceded

    def __post_init__(self) -> None:
        """Clamp values to valid ranges."""
        self.opponent_skill = max(0.0, min(1.0, self.opponent_skill))
        self.prediction_ability = max(0.0, min(1.0, self.prediction_ability))
        self.reaction_speed = max(0.0, min(1.0, self.reaction_speed))
        self.accuracy = max(0.0, min(1.0, self.accuracy))
        self.aggression = max(0.0, min(1.0, self.aggression))


@dataclass
class CurriculumMetrics:
    """Granular metrics for curriculum advancement decisions.

    Collected over an evaluation window and compared against the
    active ``DifficultyLevel``'s thresholds.
    """
    win_rate: float = 0.0
    avg_goals_scored: float = 0.0
    avg_goals_conceded: float = 0.0
    avg_rally_length: float = 0.0
    mean_reward: float = 0.0
    episodes_evaluated: int = 0

    def meets_advancement(self, level: DifficultyLevel) -> bool:
        """Check if metrics satisfy all advancement criteria."""
        return (
            self.win_rate >= level.min_win_rate_to_advance
            and self.avg_goals_scored >= level.min_avg_goals_scored
            and self.avg_goals_conceded <= level.max_avg_goals_conceded
        )


# ─────────────────────────────────────────────────────────────────────
# Difficulty progression tiers
# ─────────────────────────────────────────────────────────────────────

CURRICULUM_LEVELS: List[DifficultyLevel] = [
    DifficultyLevel(
        level=0,
        opponent_skill=0.1,
        prediction_ability=0.2,
        reaction_speed=0.05,
        accuracy=0.4,
        aggression=0.2,
        min_win_rate_to_advance=0.55,
        min_avg_goals_scored=1.0,
    ),
    DifficultyLevel(
        level=1,
        opponent_skill=0.3,
        prediction_ability=0.45,
        reaction_speed=0.10,
        accuracy=0.55,
        aggression=0.35,
        min_win_rate_to_advance=0.58,
        min_avg_goals_scored=1.5,
    ),
    DifficultyLevel(
        level=2,
        opponent_skill=0.5,
        prediction_ability=0.60,
        reaction_speed=0.15,
        accuracy=0.65,
        aggression=0.50,
        min_win_rate_to_advance=0.60,
        min_avg_goals_scored=2.0,
        max_avg_goals_conceded=5.0,
    ),
    DifficultyLevel(
        level=3,
        opponent_skill=0.65,
        prediction_ability=0.72,
        reaction_speed=0.18,
        accuracy=0.75,
        aggression=0.58,
        min_win_rate_to_advance=0.62,
        min_avg_goals_scored=2.0,
        max_avg_goals_conceded=4.0,
    ),
    DifficultyLevel(
        level=4,
        opponent_skill=0.8,
        prediction_ability=0.82,
        reaction_speed=0.22,
        accuracy=0.85,
        aggression=0.68,
        min_win_rate_to_advance=0.65,
        min_avg_goals_scored=2.5,
        max_avg_goals_conceded=3.5,
    ),
    DifficultyLevel(
        level=5,
        opponent_skill=0.9,
        prediction_ability=0.92,
        reaction_speed=0.25,
        accuracy=0.92,
        aggression=0.80,
        min_win_rate_to_advance=0.70,
        min_avg_goals_scored=3.0,
        max_avg_goals_conceded=3.0,
    ),
]
