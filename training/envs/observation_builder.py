"""
Configurable observation builder for the Air Hockey environment.

Separates observation construction from env logic (SRP). Supports both
the legacy 13-dimensional space and an extended space with opponent info.

Usage:
    builder = ObservationBuilder.standard()   # 13D legacy
    builder = ObservationBuilder.extended()   # 17D with opponent info

    obs_space = builder.get_observation_space()
    obs = builder.build(snapshot)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence

import numpy as np
from gymnasium import spaces


class ObsFeature(Enum):
    """Individual observation features — composable building blocks."""
    AI_POS = auto()           # AI mallet position (2D)
    PUCK_POS = auto()         # Puck position (2D)
    PUCK_VEL = auto()         # Puck velocity (2D)
    DISTANCE = auto()         # AI-to-puck distance (1D)
    PUCK_TO_GOAL = auto()     # Puck distance to enemy goal X (1D)
    PUCK_PROGRESS = auto()    # Puck X progress (1D)
    STEPS_SINCE_HIT = auto()  # Steps since last hit, normalized (1D)
    HEADING_FLAG = auto()     # Is puck heading toward AI? (1D)
    SCORES = auto()           # Player and AI scores normalized (2D)
    OPPONENT_POS = auto()     # Human/opponent mallet position (2D)
    OPPONENT_VEL = auto()     # Human/opponent mallet velocity (2D)
    AI_VEL = auto()           # AI mallet velocity (2D)
    GOAL_DISTANCES = auto()   # Distance to own/enemy goal (2D)


# Feature dimension map
_FEATURE_DIMS: dict[ObsFeature, int] = {
    ObsFeature.AI_POS: 2,
    ObsFeature.PUCK_POS: 2,
    ObsFeature.PUCK_VEL: 2,
    ObsFeature.DISTANCE: 1,
    ObsFeature.PUCK_TO_GOAL: 1,
    ObsFeature.PUCK_PROGRESS: 1,
    ObsFeature.STEPS_SINCE_HIT: 1,
    ObsFeature.HEADING_FLAG: 1,
    ObsFeature.SCORES: 2,
    ObsFeature.OPPONENT_POS: 2,
    ObsFeature.OPPONENT_VEL: 2,
    ObsFeature.AI_VEL: 2,
    ObsFeature.GOAL_DISTANCES: 2,
}

# Default low/high bounds per feature element
_FEATURE_BOUNDS: dict[ObsFeature, tuple[list[float], list[float]]] = {
    ObsFeature.AI_POS:          ([0, 0],   [1, 1]),
    ObsFeature.PUCK_POS:        ([0, 0],   [1, 1]),
    ObsFeature.PUCK_VEL:        ([-1, -1], [1, 1]),
    ObsFeature.DISTANCE:        ([0],      [1]),
    ObsFeature.PUCK_TO_GOAL:    ([0],      [1]),
    ObsFeature.PUCK_PROGRESS:   ([0],      [1]),
    ObsFeature.STEPS_SINCE_HIT: ([0],      [1]),
    ObsFeature.HEADING_FLAG:    ([0],      [1]),
    ObsFeature.SCORES:          ([0, 0],   [1, 1]),
    ObsFeature.OPPONENT_POS:    ([0, 0],   [1, 1]),
    ObsFeature.OPPONENT_VEL:    ([-1, -1], [1, 1]),
    ObsFeature.AI_VEL:          ([-1, -1], [1, 1]),
    ObsFeature.GOAL_DISTANCES:  ([0, 0],   [1, 1]),
}


@dataclass(frozen=True)
class ObsSnapshot:
    """Immutable data bundle consumed by ObservationBuilder.

    All raw (un-normalized) values; the builder handles normalization.
    """
    width: float
    height: float
    ai_pos: tuple[float, float]
    ai_vel: tuple[float, float]
    puck_pos: tuple[float, float]
    puck_vel: tuple[float, float]
    puck_max_speed: float
    opponent_pos: tuple[float, float]
    opponent_vel: tuple[float, float]
    steps_since_last_hit: int
    player_score: int
    ai_score: int
    score_limit: int


class ObservationBuilder:
    """Composable observation builder for Air Hockey environments.

    Construct via factory methods or by passing a list of ``ObsFeature``
    to customize exactly which signals the agent receives.
    """

    def __init__(self, features: Sequence[ObsFeature]) -> None:
        self._features = list(features)
        self._dim = sum(_FEATURE_DIMS[f] for f in self._features)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def standard(cls) -> "ObservationBuilder":
        """Legacy 13-dimensional observation space (v2 compatible)."""
        return cls([
            ObsFeature.AI_POS,
            ObsFeature.PUCK_POS,
            ObsFeature.PUCK_VEL,
            ObsFeature.DISTANCE,
            ObsFeature.PUCK_TO_GOAL,
            ObsFeature.PUCK_PROGRESS,
            ObsFeature.STEPS_SINCE_HIT,
            ObsFeature.HEADING_FLAG,
            ObsFeature.SCORES,
        ])

    @classmethod
    def extended(cls) -> "ObservationBuilder":
        """Extended 17-dimensional observation space.

        Adds opponent position & velocity (4D) to the standard space.
        Gives the agent awareness of the opponent's movement, enabling
        better anticipation and strategic play.
        """
        return cls([
            ObsFeature.AI_POS,
            ObsFeature.PUCK_POS,
            ObsFeature.PUCK_VEL,
            ObsFeature.DISTANCE,
            ObsFeature.PUCK_TO_GOAL,
            ObsFeature.PUCK_PROGRESS,
            ObsFeature.STEPS_SINCE_HIT,
            ObsFeature.HEADING_FLAG,
            ObsFeature.SCORES,
            ObsFeature.OPPONENT_POS,
            ObsFeature.OPPONENT_VEL,
        ])

    @classmethod
    def full(cls) -> "ObservationBuilder":
        """Full 21-dimensional observation space.

        Includes everything: standard + opponent info + AI velocity +
        goal distances. Maximum information for the agent.
        """
        return cls([
            ObsFeature.AI_POS,
            ObsFeature.PUCK_POS,
            ObsFeature.PUCK_VEL,
            ObsFeature.DISTANCE,
            ObsFeature.PUCK_TO_GOAL,
            ObsFeature.PUCK_PROGRESS,
            ObsFeature.STEPS_SINCE_HIT,
            ObsFeature.HEADING_FLAG,
            ObsFeature.SCORES,
            ObsFeature.OPPONENT_POS,
            ObsFeature.OPPONENT_VEL,
            ObsFeature.AI_VEL,
            ObsFeature.GOAL_DISTANCES,
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Total observation dimensionality."""
        return self._dim

    @property
    def features(self) -> list[ObsFeature]:
        """Ordered list of active features."""
        return list(self._features)

    def get_observation_space(self) -> spaces.Box:
        """Build the Gymnasium ``Box`` observation space."""
        lows: list[float] = []
        highs: list[float] = []
        for f in self._features:
            lo, hi = _FEATURE_BOUNDS[f]
            lows.extend(lo)
            highs.extend(hi)
        return spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
            dtype=np.float32,
        )

    def build(self, snap: ObsSnapshot) -> np.ndarray:
        """Convert a raw snapshot into a normalized observation array."""
        parts: list[float] = []
        W, H = snap.width, snap.height
        diag = math.hypot(W, H)
        max_speed = max(snap.puck_max_speed, 1e-6)

        for feature in self._features:
            if feature == ObsFeature.AI_POS:
                parts.extend([snap.ai_pos[0] / W, snap.ai_pos[1] / H])

            elif feature == ObsFeature.PUCK_POS:
                parts.extend([snap.puck_pos[0] / W, snap.puck_pos[1] / H])

            elif feature == ObsFeature.PUCK_VEL:
                parts.extend([
                    max(-1.0, min(1.0, snap.puck_vel[0] / max_speed)),
                    max(-1.0, min(1.0, snap.puck_vel[1] / max_speed)),
                ])

            elif feature == ObsFeature.DISTANCE:
                dist = math.hypot(
                    snap.puck_pos[0] - snap.ai_pos[0],
                    snap.puck_pos[1] - snap.ai_pos[1],
                )
                parts.append(dist / diag)

            elif feature == ObsFeature.PUCK_TO_GOAL:
                parts.append((W - snap.puck_pos[0]) / W)

            elif feature == ObsFeature.PUCK_PROGRESS:
                parts.append(snap.puck_pos[0] / W)

            elif feature == ObsFeature.STEPS_SINCE_HIT:
                parts.append(min(snap.steps_since_last_hit / 100.0, 1.0))

            elif feature == ObsFeature.HEADING_FLAG:
                parts.append(1.0 if snap.puck_vel[0] < 0 else 0.0)

            elif feature == ObsFeature.SCORES:
                sl = max(snap.score_limit, 1)
                parts.extend([
                    snap.player_score / float(sl),
                    snap.ai_score / float(sl),
                ])

            elif feature == ObsFeature.OPPONENT_POS:
                parts.extend([snap.opponent_pos[0] / W, snap.opponent_pos[1] / H])

            elif feature == ObsFeature.OPPONENT_VEL:
                # Normalize against ai_move_amount (5 px/step)
                move_amt = 5.0
                parts.extend([
                    max(-1.0, min(1.0, snap.opponent_vel[0] / move_amt)),
                    max(-1.0, min(1.0, snap.opponent_vel[1] / move_amt)),
                ])

            elif feature == ObsFeature.AI_VEL:
                move_amt = 5.0
                parts.extend([
                    max(-1.0, min(1.0, snap.ai_vel[0] / move_amt)),
                    max(-1.0, min(1.0, snap.ai_vel[1] / move_amt)),
                ])

            elif feature == ObsFeature.GOAL_DISTANCES:
                # Normalized distance from AI mallet to own goal (right side)
                # and to enemy goal (left side = x=0)
                own_goal_dist = (W - snap.ai_pos[0]) / W
                enemy_goal_dist = snap.ai_pos[0] / W
                parts.extend([own_goal_dist, enemy_goal_dist])

        return np.array(parts, dtype=np.float32)

    def describe(self) -> list[str]:
        """Return human-readable names for each dimension — useful for logging."""
        names: list[str] = []
        for f in self._features:
            dim = _FEATURE_DIMS[f]
            base = f.name.lower()
            if dim == 1:
                names.append(base)
            elif dim == 2:
                names.extend([f"{base}_x", f"{base}_y"])
            else:
                names.extend([f"{base}_{i}" for i in range(dim)])
        return names

    def __repr__(self) -> str:
        feat_str = ", ".join(f.name for f in self._features)
        return f"ObservationBuilder({self._dim}D: {feat_str})"
