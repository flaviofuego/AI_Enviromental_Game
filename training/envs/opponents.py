"""
Modular opponent system for RL training environments.

Design principles:
- **Separation of Concerns**: Each strategy is an independent, testable unit.
- **Composition over inheritance**: ``AlgorithmicOpponent`` composes strategies.
- **Extensibility**: New strategies plug in via the ``OpponentStrategy`` protocol.
- **Performance**: Pre-computed constants; zero allocations in the update loop.
- **Reusability**: Pure-logic module — no pygame dependency beyond position updates.
- **Curriculum-friendly**: All difficulty knobs exposed via ``OpponentParams``.

Architecture::

    OpponentBase (Protocol)
    ├── AlgorithmicOpponent          ← main implementation
    │   ├── OpponentState (FSM)
    │   └── Composed strategies:
    │       ├── DefensiveStrategy    ← block / track puck
    │       ├── OffensiveStrategy    ← advance & shoot at goal
    │       ├── InterceptStrategy    ← predict trajectory & intercept
    │       └── PositionalStrategy   ← hold home position
    └── SelfPlayOpponent (stub)      ← future: load PPO model as opponent
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from training.configs.curriculum import DifficultyLevel


# ─────────────────────────────────────────────────────────────────────
# Field snapshot — immutable data passed to strategies each step
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FieldSnapshot:
    """Immutable view of the field, consumed by every strategy.

    All positions are in pixel space matching ``GameConfig`` dimensions.
    The opponent controls the **left-side** mallet (player side in training).
    """

    # Field dimensions
    width: int
    height: int
    half_width: int  # pre-computed w // 2

    # Opponent (left-side) mallet
    mallet_pos: tuple[float, float]
    mallet_radius: float

    # Puck state
    puck_pos: tuple[float, float]
    puck_vel: tuple[float, float]
    puck_radius: float

    # Goal geometry (left goal — opponent's own goal)
    own_goal_x: float        # x position of own goal (0)
    own_goal_center_y: float  # center y of own goal
    goal_y1: float            # top of goal mouth
    goal_y2: float            # bottom of goal mouth

    # Rival goal (right side — AI's goal)
    rival_goal_x: float
    rival_goal_center_y: float

    @property
    def puck_in_own_half(self) -> bool:
        """Puck is in the opponent's (left) half."""
        return self.puck_pos[0] < self.half_width

    @property
    def puck_heading_toward_own_goal(self) -> bool:
        """Puck velocity has a negative x component (moving left)."""
        return self.puck_vel[0] < -0.3

    @property
    def puck_speed(self) -> float:
        return math.hypot(self.puck_vel[0], self.puck_vel[1])

    @property
    def distance_to_puck(self) -> float:
        return math.hypot(
            self.puck_pos[0] - self.mallet_pos[0],
            self.puck_pos[1] - self.mallet_pos[1],
        )


# ─────────────────────────────────────────────────────────────────────
# Difficulty parameters — single knob for curriculum control
# ─────────────────────────────────────────────────────────────────────

@dataclass
class OpponentParams:
    """Tunable difficulty parameters.

    All values range [0.0, 1.0] where 0 = easiest, 1 = hardest.
    ``from_skill`` converts a single ``skill`` scalar into a balanced param set.
    """

    skill: float = 0.3

    # Derived — set by ``from_skill``
    prediction_ability: float = 0.0
    reaction_speed: float = 0.0
    accuracy: float = 0.0
    aggression: float = 0.0
    max_speed: float = 0.0
    shot_power: float = 0.0

    # Variability — per-episode randomization range for anti-overfitting
    style_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.style_weights:
            self._derive_from_skill()

    def _derive_from_skill(self) -> None:
        s = self.skill
        self.prediction_ability = 0.2 + 0.8 * s
        self.reaction_speed = 0.05 + 0.25 * s
        self.accuracy = 0.4 + 0.6 * s
        self.aggression = 0.2 + 0.7 * s
        self.max_speed = 8.0 + 7.0 * s           # 8 – 15 px/step
        self.shot_power = 0.4 + 0.6 * s

        # Style distribution: probability of choosing each strategy
        # Low skill → mostly positional; high skill → balanced with aggression
        defensive_w = 0.35 - 0.15 * s   # 0.35 → 0.20
        offensive_w = 0.15 + 0.25 * s   # 0.15 → 0.40
        intercept_w = 0.10 + 0.20 * s   # 0.10 → 0.30
        positional_w = 0.40 - 0.30 * s  # 0.40 → 0.10
        total = defensive_w + offensive_w + intercept_w + positional_w
        self.style_weights = {
            "defensive": defensive_w / total,
            "offensive": offensive_w / total,
            "intercept": intercept_w / total,
            "positional": positional_w / total,
        }

    @classmethod
    def from_skill(cls, skill: float) -> OpponentParams:
        """Factory: create params from a single skill value [0, 1]."""
        return cls(skill=max(0.0, min(1.0, skill)))

    @classmethod
    def from_curriculum_level(cls, level: "DifficultyLevel") -> OpponentParams:  # noqa: F821
        """Factory: create params from a ``DifficultyLevel`` dataclass."""
        params = cls(skill=level.opponent_skill)
        params.prediction_ability = level.prediction_ability
        params.reaction_speed = level.reaction_speed
        params.accuracy = level.accuracy
        params.aggression = level.aggression
        return params


# ─────────────────────────────────────────────────────────────────────
# Strategy protocol — each behavior module implements this
# ─────────────────────────────────────────────────────────────────────

@runtime_checkable
class OpponentStrategy(Protocol):
    """Interface for an opponent behavior strategy."""

    name: str

    def compute_target(
        self,
        snapshot: FieldSnapshot,
        params: OpponentParams,
    ) -> tuple[float, float]:
        """Return the target (x, y) that the mallet should move toward."""
        ...


# ─────────────────────────────────────────────────────────────────────
# State machine — determines which strategy is active
# ─────────────────────────────────────────────────────────────────────

class OpponentState(Enum):
    """Finite-state machine for opponent behavior selection."""
    DEFENSIVE = auto()    # Puck heading toward own goal
    OFFENSIVE = auto()    # Puck in own half, can attack
    INTERCEPT = auto()    # Puck predictable, move to intercept
    POSITIONAL = auto()   # Puck far away, hold position


# ─────────────────────────────────────────────────────────────────────
# STRATEGY: Defensive — block puck from reaching own goal
# ─────────────────────────────────────────────────────────────────────

class DefensiveStrategy:
    """Position between puck and own goal to block shots.

    Tracks puck Y with prediction error based on skill.
    Stays near the goal line when threat is high.
    """

    name = "defensive"

    def compute_target(
        self,
        snapshot: FieldSnapshot,
        params: OpponentParams,
    ) -> tuple[float, float]:
        # Defensive X: stay between 10% and 25% of field width
        defensive_x = snapshot.width * (0.10 + 0.15 * (1.0 - params.aggression))
        defensive_x = max(snapshot.mallet_radius, defensive_x)

        # Track puck Y with prediction
        if abs(snapshot.puck_vel[0]) > 0.5:
            time_to_reach = abs(
                (snapshot.mallet_pos[0] - snapshot.puck_pos[0])
                / max(0.5, abs(snapshot.puck_vel[0]))
            )
            predicted_y = snapshot.puck_pos[1] + snapshot.puck_vel[1] * time_to_reach

            # Bounce prediction (simplified single bounce)
            if predicted_y < snapshot.puck_radius:
                predicted_y = abs(predicted_y)
            elif predicted_y > snapshot.height - snapshot.puck_radius:
                predicted_y = 2 * snapshot.height - predicted_y

            # Add prediction error inversely proportional to skill
            error_scale = (1.0 - params.prediction_ability) * snapshot.height * 0.15
            predicted_y += random.gauss(0, error_scale) if error_scale > 1 else 0

            target_y = max(snapshot.mallet_radius,
                           min(predicted_y, snapshot.height - snapshot.mallet_radius))
        else:
            # Puck slow/stationary — track Y directly
            target_y = snapshot.puck_pos[1]

        # Clamp to goal mouth range when puck is approaching fast
        if snapshot.puck_heading_toward_own_goal and snapshot.puck_speed > 3.0:
            target_y = max(snapshot.goal_y1, min(target_y, snapshot.goal_y2))

        return (defensive_x, target_y)


# ─────────────────────────────────────────────────────────────────────
# STRATEGY: Offensive — advance and shoot toward rival goal
# ─────────────────────────────────────────────────────────────────────

class OffensiveStrategy:
    """Move toward the puck and position for a shot at the rival goal.

    When close enough, aim at the rival goal center with accuracy-based
    angular error and accelerate toward the puck.
    """

    name = "offensive"

    def compute_target(
        self,
        snapshot: FieldSnapshot,
        params: OpponentParams,
    ) -> tuple[float, float]:
        dist = snapshot.distance_to_puck

        if dist < 80 * params.aggression + 30:
            # Close enough to attack — aim at rival goal with offset
            angle_to_goal = math.atan2(
                snapshot.rival_goal_center_y - snapshot.puck_pos[1],
                snapshot.rival_goal_x - snapshot.puck_pos[0],
            )
            # Apply accuracy-based angular error
            angle_error = (1.0 - params.accuracy) * random.gauss(0, 0.6)
            aim_angle = angle_to_goal + angle_error

            # Target is slightly behind the puck (to push it)
            offset = snapshot.mallet_radius + snapshot.puck_radius
            target_x = snapshot.puck_pos[0] - math.cos(aim_angle) * offset
            target_y = snapshot.puck_pos[1] - math.sin(aim_angle) * offset
        else:
            # Approach puck aggressively
            target_x = snapshot.puck_pos[0] - snapshot.mallet_radius
            target_y = snapshot.puck_pos[1]

        # Clamp to left half
        target_x = max(snapshot.mallet_radius,
                       min(target_x, snapshot.half_width - snapshot.mallet_radius))
        target_y = max(snapshot.mallet_radius,
                       min(target_y, snapshot.height - snapshot.mallet_radius))

        return (target_x, target_y)


# ─────────────────────────────────────────────────────────────────────
# STRATEGY: Intercept — predict trajectory and meet the puck
# ─────────────────────────────────────────────────────────────────────

class InterceptStrategy:
    """Predict puck trajectory and move to the intercept point.

    Uses linear extrapolation with wall bounces to find where the puck
    will cross the opponent's comfortable intercept zone.
    """

    name = "intercept"

    # Intercept zone: 15-40% of field width from left
    _ZONE_MIN_FRAC = 0.12
    _ZONE_MAX_FRAC = 0.40

    def compute_target(
        self,
        snapshot: FieldSnapshot,
        params: OpponentParams,
    ) -> tuple[float, float]:
        # If puck is barely moving, fall back to tracking
        if snapshot.puck_speed < 1.0:
            return (snapshot.width * 0.25, snapshot.puck_pos[1])

        # Where will the puck be when it crosses the intercept zone?
        intercept_x = snapshot.width * (
            self._ZONE_MIN_FRAC + (self._ZONE_MAX_FRAC - self._ZONE_MIN_FRAC) * params.aggression
        )

        # Only intercept if puck is heading toward us or is in our half
        if snapshot.puck_vel[0] >= 0 and not snapshot.puck_in_own_half:
            # Puck going away — go to default position
            return (snapshot.width * 0.20, snapshot.height / 2)

        # Time to reach intercept_x
        dx = snapshot.puck_pos[0] - intercept_x
        if abs(snapshot.puck_vel[0]) < 0.3:
            time_steps = abs(dx) / 1.0  # slow approach
        else:
            time_steps = abs(dx / snapshot.puck_vel[0])

        time_steps = min(time_steps, 120)  # cap prediction horizon

        # Predict Y with bounces
        predicted_y = self._predict_y_with_bounces(
            snapshot.puck_pos[1],
            snapshot.puck_vel[1],
            time_steps,
            snapshot.puck_radius,
            snapshot.height,
        )

        # Add prediction error
        error = (1.0 - params.prediction_ability) * random.gauss(0, snapshot.height * 0.1)
        predicted_y += error

        target_x = max(snapshot.mallet_radius,
                       min(intercept_x, snapshot.half_width - snapshot.mallet_radius))
        target_y = max(snapshot.mallet_radius,
                       min(predicted_y, snapshot.height - snapshot.mallet_radius))

        return (target_x, target_y)

    @staticmethod
    def _predict_y_with_bounces(
        y: float, vy: float, steps: float,
        radius: float, height: float,
    ) -> float:
        """Simulate Y position after ``steps`` with wall bounces."""
        for _ in range(int(min(steps, 60))):
            y += vy
            if y - radius < 0:
                y = radius
                vy = abs(vy)
            elif y + radius > height:
                y = height - radius
                vy = -abs(vy)
        return y


# ─────────────────────────────────────────────────────────────────────
# STRATEGY: Positional — hold home position with slight tracking
# ─────────────────────────────────────────────────────────────────────

class PositionalStrategy:
    """Maintain a safe home position with reactive Y tracking.

    Used when the puck is far away. Keeps the mallet near the goal
    center while gently tracking puck Y.
    """

    name = "positional"

    def compute_target(
        self,
        snapshot: FieldSnapshot,
        params: OpponentParams,
    ) -> tuple[float, float]:
        # Home X: 20-25% from left edge
        home_x = snapshot.width * 0.22

        # Home Y: blend between center and puck Y
        center_y = snapshot.height / 2
        tracking_strength = 0.3 * params.prediction_ability
        home_y = center_y + (snapshot.puck_pos[1] - center_y) * tracking_strength

        # Bias toward goal mouth when puck is in the other half
        if not snapshot.puck_in_own_half:
            home_y = center_y + (home_y - center_y) * 0.5

        home_y = max(snapshot.mallet_radius,
                     min(home_y, snapshot.height - snapshot.mallet_radius))

        return (home_x, home_y)


# ─────────────────────────────────────────────────────────────────────
# Main opponent — composes strategies via FSM
# ─────────────────────────────────────────────────────────────────────

class AlgorithmicOpponent:
    """State-machine-driven opponent that composes independent strategies.

    Usage::

        opponent = AlgorithmicOpponent(OpponentParams.from_skill(0.5))
        # Each step:
        new_x, new_y, vx, vy = opponent.update(snapshot)
    """

    def __init__(self, params: OpponentParams | None = None) -> None:
        self._params = params or OpponentParams.from_skill(0.3)
        self._state = OpponentState.POSITIONAL

        # Compose strategies
        self._strategies: dict[OpponentState, OpponentStrategy] = {
            OpponentState.DEFENSIVE: DefensiveStrategy(),
            OpponentState.OFFENSIVE: OffensiveStrategy(),
            OpponentState.INTERCEPT: InterceptStrategy(),
            OpponentState.POSITIONAL: PositionalStrategy(),
        }

        # Reaction delay buffer — smooths transitions
        self._reaction_frames: int = 0
        self._reaction_cooldown: int = 0

        # Per-episode style bias (randomized in ``reset_episode``)
        self._style_bias: str = "balanced"

        self.reset_episode()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def params(self) -> OpponentParams:
        return self._params

    @params.setter
    def params(self, value: OpponentParams) -> None:
        self._params = value
        # Recalculate reaction cooldown for new skill level
        self._reaction_cooldown = max(0, int((1.0 - value.skill) * 6))

    @property
    def state(self) -> OpponentState:
        return self._state

    def reset_episode(self) -> None:
        """Randomize per-episode style to prevent overfitting."""
        self._state = OpponentState.POSITIONAL
        self._reaction_frames = 0

        # Weighted random style bias
        weights = self._params.style_weights
        styles = list(weights.keys())
        probs = [weights[s] for s in styles]
        self._style_bias = random.choices(styles, weights=probs, k=1)[0]

    def update(
        self,
        snapshot: FieldSnapshot,
        current_pos: tuple[float, float],
    ) -> tuple[float, float, float, float]:
        """Compute new mallet position and velocity.

        Returns:
            ``(new_x, new_y, velocity_x, velocity_y)``
        """
        # 1. Determine state
        new_state = self._determine_state(snapshot)
        if new_state != self._state:
            self._reaction_frames = self._reaction_cooldown
            self._state = new_state

        # 2. Apply reaction delay — hold current position during transition
        if self._reaction_frames > 0:
            self._reaction_frames -= 1
            return (*current_pos, 0.0, 0.0)

        # 3. Get target from active strategy
        strategy = self._strategies[self._state]
        target_x, target_y = strategy.compute_target(snapshot, self._params)

        # 4. Move toward target with speed limits and noise
        new_x, new_y = self._move_toward(
            current_pos, (target_x, target_y), snapshot
        )

        # 5. Compute velocity for collision physics
        vx = new_x - current_pos[0]
        vy = new_y - current_pos[1]

        return (new_x, new_y, vx, vy)

    # ------------------------------------------------------------------
    # FSM — state determination
    # ------------------------------------------------------------------

    def _determine_state(self, snapshot: FieldSnapshot) -> OpponentState:
        """Select behavior state based on field conditions and style bias."""

        # High-priority: puck heading directly at our goal fast
        if (snapshot.puck_heading_toward_own_goal
                and snapshot.puck_speed > 2.0
                and snapshot.puck_in_own_half):
            return OpponentState.DEFENSIVE

        # Puck heading toward us — intercept if we have the skill
        if snapshot.puck_heading_toward_own_goal and snapshot.puck_speed > 1.0:
            if self._style_bias == "intercept" or self._params.prediction_ability > 0.5:
                return OpponentState.INTERCEPT
            return OpponentState.DEFENSIVE

        # Puck in our half and slow/controllable — attack
        if snapshot.puck_in_own_half:
            if self._style_bias == "defensive" and snapshot.distance_to_puck > 100:
                return OpponentState.POSITIONAL
            return OpponentState.OFFENSIVE

        # Puck in rival half — positional or defensive based on bias
        if self._style_bias == "offensive" and self._params.aggression > 0.5:
            return OpponentState.POSITIONAL  # ready to pounce
        return OpponentState.POSITIONAL

    # ------------------------------------------------------------------
    # Movement — smooth approach with noise
    # ------------------------------------------------------------------

    def _move_toward(
        self,
        current: tuple[float, float],
        target: tuple[float, float],
        snapshot: FieldSnapshot,
    ) -> tuple[float, float]:
        """Move toward target with reaction speed and noise."""
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        dist = math.hypot(dx, dy)

        if dist < 0.5:
            return current

        # Scale movement by reaction speed
        speed = self._params.max_speed * self._params.reaction_speed
        # Boost speed when puck is close and heading toward us (urgency)
        if (snapshot.puck_heading_toward_own_goal
                and snapshot.distance_to_puck < snapshot.width * 0.3):
            speed = self._params.max_speed * min(1.0, self._params.reaction_speed * 2.5)

        if dist > speed:
            scale = speed / dist
            dx *= scale
            dy *= scale

        new_x = current[0] + dx
        new_y = current[1] + dy

        # Add movement noise (decreases with skill)
        noise_scale = (1.0 - self._params.skill) * 2.0
        if noise_scale > 0.1:
            new_x += random.gauss(0, noise_scale)
            new_y += random.gauss(0, noise_scale)

        # Clamp to left half
        r = snapshot.mallet_radius
        new_x = max(r, min(new_x, snapshot.half_width - r))
        new_y = max(r, min(new_y, snapshot.height - r))

        return (new_x, new_y)


# ─────────────────────────────────────────────────────────────────────
# Self-play stub (future extensibility)
# ─────────────────────────────────────────────────────────────────────

class SelfPlayOpponent:
    """Stub for a self-play opponent that loads a trained PPO model.

    NOT IMPLEMENTED — placeholder for future extensibility.
    When implemented, it would:
    1. Load a ``PPO.load(path)`` model.
    2. Mirror observations (swap left/right).
    3. Convert model actions to movement deltas.
    """

    name = "self_play"

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._model = None  # Loaded lazily

    def update(
        self,
        snapshot: FieldSnapshot,
        current_pos: tuple[float, float],
    ) -> tuple[float, float, float, float]:
        raise NotImplementedError(
            "SelfPlayOpponent is a stub for future implementation. "
            "Use AlgorithmicOpponent for training."
        )

    def reset_episode(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────
# Factory — create opponents from different config sources
# ─────────────────────────────────────────────────────────────────────

class OpponentFactory:
    """Create opponent instances from various configuration sources."""

    @staticmethod
    def from_skill(skill: float) -> AlgorithmicOpponent:
        """Create an opponent with a single skill value [0, 1]."""
        return AlgorithmicOpponent(OpponentParams.from_skill(skill))

    @staticmethod
    def from_curriculum_level(level: "DifficultyLevel") -> AlgorithmicOpponent:  # noqa: F821
        """Create an opponent from a ``DifficultyLevel`` dataclass."""
        return AlgorithmicOpponent(OpponentParams.from_curriculum_level(level))

    @staticmethod
    def from_params(params: OpponentParams) -> AlgorithmicOpponent:
        """Create an opponent from explicit parameters."""
        return AlgorithmicOpponent(params)
