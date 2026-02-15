"""Curriculum learning configuration for progressive difficulty training."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DifficultyLevel:
    """A single difficulty tier."""
    level: int
    opponent_skill: float
    prediction_ability: float
    reaction_speed: float
    accuracy: float
    aggression: float
    min_win_rate_to_advance: float = 0.6


# Difficulty progression tiers
CURRICULUM_LEVELS: List[DifficultyLevel] = [
    DifficultyLevel(0, opponent_skill=0.1, prediction_ability=0.3, reaction_speed=0.05, accuracy=0.5, aggression=0.3),
    DifficultyLevel(1, opponent_skill=0.3, prediction_ability=0.5, reaction_speed=0.10, accuracy=0.6, aggression=0.4),
    DifficultyLevel(2, opponent_skill=0.5, prediction_ability=0.6, reaction_speed=0.15, accuracy=0.7, aggression=0.5),
    DifficultyLevel(3, opponent_skill=0.7, prediction_ability=0.7, reaction_speed=0.18, accuracy=0.8, aggression=0.6),
    DifficultyLevel(4, opponent_skill=0.8, prediction_ability=0.8, reaction_speed=0.20, accuracy=0.85, aggression=0.7),
    DifficultyLevel(5, opponent_skill=0.9, prediction_ability=0.9, reaction_speed=0.25, accuracy=0.9, aggression=0.8),
]
