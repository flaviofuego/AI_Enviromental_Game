"""
Componentized reward system for Air Hockey RL training.

Design principles:
- **Separation of Concerns**: Each reward component is an independent calculator
- **PBRS (Potential-Based Reward Shaping)**: Shaping rewards use Φ(s') - Φ(s)
  to preserve optimal policy (Ng et al., 1999)
- **Offensive:Defensive ratio ~2:1–3:1**: Prevents passive policies
- **Anti-exploit**: Diminishing returns on repetitive actions, caps per episode
- **Extensibility**: New components added without modifying existing ones

Architecture:
    RewardCalculator (orchestrator)
    ├── GoalRewardComponent         — sparse, event-based (+/- for goals)
    ├── HitRewardComponent          — event-based (hit quality, direction)
    ├── ShotDirectionComponent      — event-based (NUEVO: post-hit direction)
    ├── ClearRewardComponent        — event-based (NUEVO: defensive clear)
    ├── PositionalRewardComponent   — dense, PBRS (gap control, coverage)
    ├── DefensiveRewardComponent    — dense, PBRS (NUEVO: blocking, proximity)
    ├── PressureRewardComponent     — dense (NUEVO: goal pressure)
    └── DisciplineComponent         — penalties (inactivity, net-front)
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, runtime_checkable

from shared.physics import dot_product, normalize_vector, vector_length, calculate_vector


# ─────────────────────────────────────────────────────────────────────
# State snapshot — immutable data passed to every component each step
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FieldState:
    """Immutable snapshot of the field for reward computation.

    All positions/velocities are in pixel-space (same as base_env).
    """
    # Field dimensions
    width: float
    height: float

    # AI mallet
    ai_pos: tuple[float, float]
    ai_vel: tuple[float, float]

    # Puck current
    puck_pos: tuple[float, float]
    puck_vel: tuple[float, float]
    puck_speed: float  # pre-computed magnitude
    puck_max_speed: float

    # Puck BEFORE this step's collision (for shot direction analysis)
    puck_vel_pre_hit: tuple[float, float] | None

    # Distances
    ai_puck_distance: float
    ai_puck_prev_distance: float

    # Mallet radius for safe-distance checks
    ai_mallet_radius: float

    # Events this step
    ai_hit_puck: bool
    goal: str | None  # "player" | "ai" | None

    # Derived booleans (computed once, shared)
    puck_in_ai_half: bool
    puck_heading_toward_ai: bool  # puck vx > 0

    # Score context
    player_score: int
    ai_score: int
    score_limit: int

    # Step counters
    steps_since_last_hit: int
    consecutive_hits: int


# ─────────────────────────────────────────────────────────────────────
# Reward breakdown — for logging / debugging
# ─────────────────────────────────────────────────────────────────────

class RewardCategory(Enum):
    """Categories for reward breakdown tracking."""
    GOAL = auto()
    HIT = auto()
    SHOT_DIRECTION = auto()
    CLEAR = auto()
    POSITIONAL = auto()
    DEFENSIVE = auto()
    PRESSURE = auto()
    DISCIPLINE = auto()
    SHAPING = auto()


@dataclass
class RewardBreakdown:
    """Detailed breakdown of each reward component's contribution."""
    components: dict[str, float] = field(default_factory=dict)
    categories: dict[RewardCategory, float] = field(default_factory=dict)
    total: float = 0.0

    def add(self, name: str, value: float, category: RewardCategory) -> None:
        if abs(value) < 1e-8:
            return
        self.components[name] = self.components.get(name, 0.0) + value
        self.categories[category] = self.categories.get(category, 0.0) + value
        self.total += value


# ─────────────────────────────────────────────────────────────────────
# Component protocol — each reward module implements this
# ─────────────────────────────────────────────────────────────────────

@runtime_checkable
class RewardComponent(Protocol):
    """Interface for a single reward component."""

    name: str

    def calculate(self, state: FieldState, breakdown: RewardBreakdown) -> None:
        """Compute and add this component's reward contribution."""
        ...

    def reset(self) -> None:
        """Reset internal state at episode start."""
        ...


# ─────────────────────────────────────────────────────────────────────
# COMPONENT: Goal reward (sparse, event-based)
# ─────────────────────────────────────────────────────────────────────

class GoalRewardComponent:
    """Sparse reward for goals scored and conceded.

    Values follow the plan's rubric:
    - AI scores:      +5.0
    - AI concedes:    -4.0 base, -1.0 extra for negligence (far from goal)
    """

    name = "goal"

    # Negligence threshold: if AI is in the left 60% of its half when
    # conceding, it was positionally negligent.
    _NEGLIGENCE_THRESHOLD = 0.6  # fraction of field width

    def calculate(self, state: FieldState, breakdown: RewardBreakdown) -> None:
        if state.goal == "ai":
            breakdown.add("goal_scored", 5.0, RewardCategory.GOAL)
        elif state.goal == "player":
            breakdown.add("goal_conceded", -4.0, RewardCategory.GOAL)
            # Negligence: AI was too far from its own goal (right side)
            own_goal_x = state.width
            dist_to_own_goal = own_goal_x - state.ai_pos[0]
            field_half = state.width / 2
            if dist_to_own_goal > field_half * self._NEGLIGENCE_THRESHOLD:
                breakdown.add("negligence_penalty", -1.0, RewardCategory.GOAL)

    def reset(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────
# COMPONENT: Hit reward (event-based)
# ─────────────────────────────────────────────────────────────────────

class HitRewardComponent:
    """Reward for hitting the puck, scaled by shot quality.

    - Base hit: +0.8
    - Shot quality: +2.5 * alignment * speed_ratio (when aimed at goal)
    - Hard shot bonus: +0.5 when speed > 0.6 * max_speed
    - Diminishing returns: consecutive hits without direction change
      reduce base reward by 20% each (anti-exploit: "hit farming")
    """

    name = "hit"
    _MAX_CONSECUTIVE_DISCOUNT = 5  # After 5 consecutive, base = ~0.8 * 0.8^5 ≈ 0.26

    def __init__(self) -> None:
        self._consecutive_hits = 0

    def calculate(self, state: FieldState, breakdown: RewardBreakdown) -> None:
        if not state.ai_hit_puck:
            if state.steps_since_last_hit > 5:
                self._consecutive_hits = 0
            return

        # Diminishing base reward for consecutive hits (anti-exploit)
        discount = 0.8 ** min(self._consecutive_hits, self._MAX_CONSECUTIVE_DISCOUNT)
        base_reward = 0.8 * discount
        breakdown.add("hit_base", base_reward, RewardCategory.HIT)
        self._consecutive_hits += 1

        # Shot quality: alignment with opponent's goal (at x=0, y=H/2)
        goal_center = (0.0, state.height / 2.0)
        to_goal = calculate_vector(state.puck_pos, goal_center)
        to_goal_n = normalize_vector(to_goal)

        if state.puck_speed > 0:
            puck_dir = normalize_vector(state.puck_vel)
            alignment = dot_product(to_goal_n, puck_dir)
            speed_ratio = min(1.0, state.puck_speed / state.puck_max_speed)

            if alignment > 0:
                shot_quality = alignment * speed_ratio
                breakdown.add("shot_quality", 2.5 * shot_quality, RewardCategory.HIT)

                # Hard shot bonus
                if speed_ratio > 0.6:
                    breakdown.add("hard_shot", 0.5, RewardCategory.HIT)

    def reset(self) -> None:
        self._consecutive_hits = 0


# ─────────────────────────────────────────────────────────────────────
# COMPONENT: Shot direction (NUEVO — event-based)
# ─────────────────────────────────────────────────────────────────────

class ShotDirectionComponent:
    """Reward/penalty based on post-hit puck direction relative to enemy goal.

    Uses dot_product(puck_velocity, direction_to_enemy_goal) after hit:
    - dot > 0.7: +1.5 * dot (aimed well at goal)
    - dot < 0:   -0.5 (hit puck away from goal / toward own goal)

    Requires `puck_vel_pre_hit` to be set so we can compare direction change.
    """

    name = "shot_direction"

    def calculate(self, state: FieldState, breakdown: RewardBreakdown) -> None:
        if not state.ai_hit_puck or state.puck_speed < 0.5:
            return

        # Direction from puck to enemy goal (x=0, y=H/2)
        enemy_goal = (0.0, state.height / 2.0)
        to_goal = calculate_vector(state.puck_pos, enemy_goal)
        to_goal_n = normalize_vector(to_goal)

        puck_dir = normalize_vector(state.puck_vel)
        dp = dot_product(puck_dir, to_goal_n)

        if dp > 0.7:
            breakdown.add("shot_to_goal", 1.5 * dp, RewardCategory.SHOT_DIRECTION)
        elif dp < 0:
            breakdown.add("shot_away", -0.5, RewardCategory.SHOT_DIRECTION)

    def reset(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────
# COMPONENT: Clear reward (NUEVO — event-based, defensive)
# ─────────────────────────────────────────────────────────────────────

class ClearRewardComponent:
    """Reward for successful defensive clears.

    When puck was heading toward AI goal (vx > 0) pre-hit and
    post-hit the puck direction is reversed (vx < 0): +1.5

    This encourages the agent to actively deflect incoming threats.
    """

    name = "clear"

    def calculate(self, state: FieldState, breakdown: RewardBreakdown) -> None:
        if not state.ai_hit_puck:
            return

        pre_vel = state.puck_vel_pre_hit
        if pre_vel is None:
            return

        # Puck was heading toward AI goal (positive X for right-side goal)
        puck_was_incoming = pre_vel[0] > 1.0
        # After hit, puck reversed direction (now heading toward opponent)
        puck_now_outgoing = state.puck_vel[0] < -1.0

        if puck_was_incoming and puck_now_outgoing:
            breakdown.add("clear", 1.5, RewardCategory.CLEAR)

    def reset(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────
# COMPONENT: Interception (event-based, defensive)
# ─────────────────────────────────────────────────────────────────────

class InterceptionComponent:
    """Reward for intercepting an incoming puck.

    +1.0 when hitting a puck that was heading toward AI's goal.
    Tracks whether last hit was defensive for counterattack detection.
    """

    name = "interception"

    def __init__(self) -> None:
        self.last_hit_was_defensive = False

    def calculate(self, state: FieldState, breakdown: RewardBreakdown) -> None:
        if not state.ai_hit_puck:
            return

        puck_was_incoming = (
            state.puck_vel_pre_hit is not None and state.puck_vel_pre_hit[0] > 2.0
        ) or (
            state.ai_puck_prev_distance < 80
            and state.ai_pos[0] > state.width * 0.7
        )

        if puck_was_incoming:
            breakdown.add("interception", 1.0, RewardCategory.DEFENSIVE)
            self.last_hit_was_defensive = True
        else:
            self.last_hit_was_defensive = False

    def reset(self) -> None:
        self.last_hit_was_defensive = False


# ─────────────────────────────────────────────────────────────────────
# COMPONENT: Counterattack (event-based)
# ─────────────────────────────────────────────────────────────────────

class CounterattackComponent:
    """Reward for quick offensive transition after defensive interception.

    +0.5 when the last hit was defensive and now puck is heading toward
    opponent's goal (vx < -1).
    """

    name = "counterattack"

    def __init__(self, interception_ref: InterceptionComponent) -> None:
        self._interception = interception_ref

    def calculate(self, state: FieldState, breakdown: RewardBreakdown) -> None:
        if not state.ai_hit_puck:
            return

        if self._interception.last_hit_was_defensive and state.puck_vel[0] < -1.0:
            breakdown.add("counterattack", 0.5, RewardCategory.HIT)
            self._interception.last_hit_was_defensive = False

    def reset(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────
# COMPONENT: Positional play (dense, PBRS-inspired)
# ─────────────────────────────────────────────────────────────────────

class PositionalRewardComponent:
    """Dense positional rewards using PBRS-style computation.

    Sub-signals:
    - Gap control:   +0.05 max when at ~100px from puck (optimal distance)
    - Approach:      +0.08 when reducing distance to puck (when too far)
    - Positional:    +0.03 when between puck and own goal
    - Y-alignment:   +0.02 when vertically aligned with puck
    - Pressure play: +0.01 when near center while puck is in opponent half

    These use a potential-based structure where relevant to avoid
    degenerate policy shifts.
    """

    name = "positional"

    def __init__(self) -> None:
        self._prev_potential: float | None = None

    def calculate(self, state: FieldState, breakdown: RewardBreakdown) -> None:
        W, H = state.width, state.height

        # ── Gap Control ──
        if state.puck_in_ai_half:
            optimal_dist = 100.0
            gap_error = abs(state.ai_puck_distance - optimal_dist) / optimal_dist
            gap_reward = max(0.0, 1.0 - gap_error) * 0.05
            breakdown.add("gap_control", gap_reward, RewardCategory.POSITIONAL)

            # Approach reward when too far
            if (state.ai_puck_distance > optimal_dist * 1.5
                    and state.ai_puck_distance < state.ai_puck_prev_distance):
                approach_factor = min(
                    1.0, (state.ai_puck_prev_distance - state.ai_puck_distance) * 0.1
                )
                breakdown.add("approach", 0.08 * approach_factor, RewardCategory.POSITIONAL)

        # ── Positional Play (between puck and own goal) ──
        if state.puck_in_ai_half and state.ai_pos[0] > state.puck_pos[0]:
            coverage = 1.0 - abs(
                state.ai_pos[0] - (state.puck_pos[0] + W) / 2
            ) / (W / 2)
            breakdown.add("positional_play", 0.03 * max(0.0, coverage), RewardCategory.POSITIONAL)

        # ── Y-Axis Alignment ──
        if state.puck_in_ai_half:
            y_diff = abs(state.ai_pos[1] - state.puck_pos[1])
            y_alignment = 1.0 - min(y_diff / (H * 0.5), 1.0)
            breakdown.add("y_alignment", 0.02 * y_alignment, RewardCategory.POSITIONAL)

        # ── Pressure Play ──
        if not state.puck_in_ai_half:
            dist_to_center = abs(state.ai_pos[0] - (W * 0.6))
            center_proximity = 1.0 - min(dist_to_center / (W * 0.3), 1.0)
            breakdown.add("pressure_play", 0.01 * center_proximity, RewardCategory.POSITIONAL)

    def reset(self) -> None:
        self._prev_potential = None


# ─────────────────────────────────────────────────────────────────────
# COMPONENT: Defensive positioning (NUEVO — dense)
# ─────────────────────────────────────────────────────────────────────

class DefensiveRewardComponent:
    """Dense defensive positioning rewards.

    - Block position: +0.1 when between puck and own goal in AI half
    - Defensive proximity: +0.05 when < 100px from puck in defensive half
    """

    name = "defensive"

    def calculate(self, state: FieldState, breakdown: RewardBreakdown) -> None:
        W, H = state.width, state.height

        if not state.puck_in_ai_half:
            return

        # Only when puck is actively threatening (heading toward AI goal)
        if not state.puck_heading_toward_ai:
            return

        # ── Block Position ──
        # AI is between puck and own goal (x=W), and roughly aligned in Y
        if state.ai_pos[0] > state.puck_pos[0]:
            y_coverage = 1.0 - min(
                abs(state.ai_pos[1] - state.puck_pos[1]) / (H * 0.4), 1.0
            )
            if y_coverage > 0.3:  # Only reward meaningful blocking
                breakdown.add("block_position", 0.1 * y_coverage, RewardCategory.DEFENSIVE)

        # ── Defensive Proximity ──
        if state.ai_puck_distance < 100.0:
            proximity_factor = 1.0 - (state.ai_puck_distance / 100.0)
            breakdown.add("defensive_proximity", 0.05 * proximity_factor, RewardCategory.DEFENSIVE)

    def reset(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────
# COMPONENT: Goal pressure (NUEVO — dense)
# ─────────────────────────────────────────────────────────────────────

class GoalPressureComponent:
    """Small per-frame reward when the puck is in the opponent's half
    AND heading toward their goal.

    +0.03 per frame — incentivizes maintaining offensive pressure.
    """

    name = "goal_pressure"

    def calculate(self, state: FieldState, breakdown: RewardBreakdown) -> None:
        # Puck in opponent (left) half and heading left (toward goal at x=0)
        if state.puck_pos[0] < state.width / 2 and state.puck_vel[0] < -0.5:
            breakdown.add("goal_pressure", 0.03, RewardCategory.PRESSURE)

    def reset(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────
# COMPONENT: Discipline penalties (dense)
# ─────────────────────────────────────────────────────────────────────

class DisciplineComponent:
    """Penalties for undesirable behaviors.

    - Net-front camping:  -0.05 when too close to own goal
    - Inactivity:         -0.02 when stationary with puck nearby
    """

    name = "discipline"

    def calculate(self, state: FieldState, breakdown: RewardBreakdown) -> None:
        W = state.width

        # ── Net-Front Discipline ──
        dist_to_own_goal = W - state.ai_pos[0]
        if dist_to_own_goal < state.ai_mallet_radius * 2:
            breakdown.add("net_front", -0.05, RewardCategory.DISCIPLINE)

        # ── Inactivity ──
        movement = vector_length(state.ai_vel)
        if (movement < 0.1
                and state.puck_in_ai_half
                and state.ai_puck_distance < 200):
            breakdown.add("inactivity", -0.02, RewardCategory.DISCIPLINE)

    def reset(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────
# COMPONENT: PBRS shaping (potential-based, dense)
# ─────────────────────────────────────────────────────────────────────

class PBRSComponent:
    """Potential-Based Reward Shaping (Ng et al., 1999).

    Φ(s) = w1 * puck_progress + w2 * defensive_alignment

    F(s, s') = γ * Φ(s') - Φ(s)

    This preserves the optimal policy while accelerating convergence.
    The potential expresses "how good" a state is for the AI.
    """

    name = "pbrs"

    def __init__(self, gamma: float = 0.995,
                 w_progress: float = 0.5,
                 w_defense: float = 0.2) -> None:
        self._gamma = gamma
        self._w_progress = w_progress
        self._w_defense = w_defense
        self._prev_potential: float | None = None

    def _potential(self, state: FieldState) -> float:
        """Compute Φ(s): higher when state favors AI."""
        W, H = state.width, state.height

        # Puck progress toward opponent goal (x=0 is opponent goal)
        # Range [0, 1]: 1 = at opponent goal, 0 = at AI goal
        puck_progress = 1.0 - (state.puck_pos[0] / W)

        # Defensive alignment: AI between puck and own goal, Y-aligned
        if state.puck_in_ai_half and state.ai_pos[0] > state.puck_pos[0]:
            y_diff = abs(state.ai_pos[1] - state.puck_pos[1]) / H
            defense_alignment = max(0.0, 1.0 - y_diff * 2)
        else:
            defense_alignment = 0.0

        return (self._w_progress * puck_progress
                + self._w_defense * defense_alignment)

    def calculate(self, state: FieldState, breakdown: RewardBreakdown) -> None:
        current_potential = self._potential(state)

        if self._prev_potential is not None:
            # F = γ * Φ(s') - Φ(s)
            shaping = self._gamma * current_potential - self._prev_potential
            breakdown.add("pbrs_shaping", shaping, RewardCategory.SHAPING)

        self._prev_potential = current_potential

    def reset(self) -> None:
        self._prev_potential = None


# ─────────────────────────────────────────────────────────────────────
# Orchestrator — assembles all components
# ─────────────────────────────────────────────────────────────────────

class RewardCalculator:
    """Orchestrates all reward components.

    Usage:
        calc = RewardCalculator.default()
        # At episode start:
        calc.reset()
        # Each step:
        reward, breakdown = calc.calculate(state)
    """

    def __init__(self, components: list[RewardComponent]) -> None:
        self._components = components

    @classmethod
    def default(cls, gamma: float = 0.995) -> RewardCalculator:
        """Create calculator with all standard components.

        Returns a fully configured RewardCalculator with the v2 rubric.
        """
        interception = InterceptionComponent()
        return cls([
            GoalRewardComponent(),
            HitRewardComponent(),
            ShotDirectionComponent(),
            ClearRewardComponent(),
            interception,
            CounterattackComponent(interception),
            PositionalRewardComponent(),
            DefensiveRewardComponent(),
            GoalPressureComponent(),
            DisciplineComponent(),
            PBRSComponent(gamma=gamma),
        ])

    @classmethod
    def minimal(cls) -> RewardCalculator:
        """Sparse-only calculator (goals + hits). For ablation studies."""
        return cls([
            GoalRewardComponent(),
            HitRewardComponent(),
        ])

    def calculate(self, state: FieldState) -> tuple[float, RewardBreakdown]:
        """Run all components and return (total_reward, breakdown)."""
        breakdown = RewardBreakdown()
        for component in self._components:
            component.calculate(state, breakdown)
        return breakdown.total, breakdown

    def reset(self) -> None:
        """Reset all components at episode start."""
        for component in self._components:
            component.reset()

    @property
    def component_names(self) -> list[str]:
        """Names of active components (for logging)."""
        return [c.name for c in self._components]
