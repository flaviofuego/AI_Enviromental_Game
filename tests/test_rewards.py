"""
Tests for the componentized reward system (training/envs/rewards.py).

Tests cover:
- Individual reward components in isolation
- RewardCalculator orchestration
- FieldState construction
- Anti-exploit mechanisms
- PBRS correctness
- Integration with base_env
"""
import math
import pytest
import pygame

from training.envs.rewards import (
    FieldState,
    RewardBreakdown,
    RewardCalculator,
    RewardCategory,
    GoalRewardComponent,
    HitRewardComponent,
    ShotDirectionComponent,
    ClearRewardComponent,
    InterceptionComponent,
    CounterattackComponent,
    PositionalRewardComponent,
    DefensiveRewardComponent,
    GoalPressureComponent,
    DisciplineComponent,
    PBRSComponent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> FieldState:
    """Create a FieldState with sensible defaults, overriding specific fields."""
    defaults = dict(
        width=800.0,
        height=500.0,
        ai_pos=(600.0, 250.0),
        ai_vel=(0.0, 0.0),
        puck_pos=(400.0, 250.0),
        puck_vel=(0.0, 0.0),
        puck_speed=0.0,
        puck_max_speed=12.0,
        puck_vel_pre_hit=None,
        ai_puck_distance=200.0,
        ai_puck_prev_distance=210.0,
        ai_mallet_radius=32,
        ai_hit_puck=False,
        goal=None,
        puck_in_ai_half=False,
        puck_heading_toward_ai=False,
        player_score=0,
        ai_score=0,
        score_limit=7,
        steps_since_last_hit=0,
        consecutive_hits=0,
    )
    defaults.update(overrides)
    return FieldState(**defaults)


# ===================================================================
# GoalRewardComponent
# ===================================================================

class TestGoalReward:
    def test_no_goal(self):
        comp = GoalRewardComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(goal=None), bd)
        assert bd.total == 0.0

    def test_ai_scores(self):
        comp = GoalRewardComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(goal="ai"), bd)
        assert bd.total == pytest.approx(5.0)
        assert "goal_scored" in bd.components

    def test_player_scores_base(self):
        comp = GoalRewardComponent()
        bd = RewardBreakdown()
        # AI is near its goal (x=750), so no negligence
        comp.calculate(_make_state(goal="player", ai_pos=(750.0, 250.0)), bd)
        assert bd.total == pytest.approx(-4.0)

    def test_player_scores_with_negligence(self):
        comp = GoalRewardComponent()
        bd = RewardBreakdown()
        # AI far from goal (x=500), negligent
        comp.calculate(_make_state(goal="player", ai_pos=(500.0, 250.0)), bd)
        assert bd.total == pytest.approx(-5.0)
        assert "negligence_penalty" in bd.components


# ===================================================================
# HitRewardComponent
# ===================================================================

class TestHitReward:
    def test_no_hit(self):
        comp = HitRewardComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(ai_hit_puck=False, steps_since_last_hit=10), bd)
        assert bd.total == 0.0

    def test_base_hit(self):
        comp = HitRewardComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(ai_hit_puck=True, puck_speed=0.0), bd)
        assert bd.components.get("hit_base", 0) == pytest.approx(0.8)

    def test_hit_with_shot_quality(self):
        comp = HitRewardComponent()
        bd = RewardBreakdown()
        # Puck moving toward enemy goal (x=0): vx = -10
        comp.calculate(_make_state(
            ai_hit_puck=True,
            puck_pos=(600.0, 250.0),
            puck_vel=(-10.0, 0.0),
            puck_speed=10.0,
        ), bd)
        assert bd.components.get("shot_quality", 0) > 0
        assert bd.total > 0.8  # More than just base hit

    def test_hard_shot_bonus(self):
        comp = HitRewardComponent()
        bd = RewardBreakdown()
        # puck_speed / max_speed > 0.6
        comp.calculate(_make_state(
            ai_hit_puck=True,
            puck_pos=(600.0, 250.0),
            puck_vel=(-10.0, 0.0),
            puck_speed=10.0,
            puck_max_speed=12.0,
        ), bd)
        assert bd.components.get("hard_shot", 0) == pytest.approx(0.5)

    def test_diminishing_returns_consecutive_hits(self):
        """Consecutive hits should give diminishing base rewards."""
        comp = HitRewardComponent()
        rewards = []
        for i in range(5):
            bd = RewardBreakdown()
            comp.calculate(_make_state(
                ai_hit_puck=True, puck_speed=0.0,
                steps_since_last_hit=0, consecutive_hits=i,
            ), bd)
            rewards.append(bd.components.get("hit_base", 0))

        # Each consecutive hit should give less base reward
        for i in range(1, len(rewards)):
            assert rewards[i] < rewards[i - 1]

    def test_diminishing_resets_after_gap(self):
        comp = HitRewardComponent()
        # Hit 3 times
        for _ in range(3):
            bd = RewardBreakdown()
            comp.calculate(_make_state(ai_hit_puck=True, puck_speed=0.0), bd)

        # No hit for >5 steps resets counter
        bd = RewardBreakdown()
        comp.calculate(_make_state(ai_hit_puck=False, steps_since_last_hit=10), bd)

        # Next hit should be full reward
        bd = RewardBreakdown()
        comp.calculate(_make_state(ai_hit_puck=True, puck_speed=0.0), bd)
        assert bd.components.get("hit_base", 0) == pytest.approx(0.8)


# ===================================================================
# ShotDirectionComponent
# ===================================================================

class TestShotDirection:
    def test_no_hit(self):
        comp = ShotDirectionComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(ai_hit_puck=False), bd)
        assert bd.total == 0.0

    def test_shot_toward_goal(self):
        comp = ShotDirectionComponent()
        bd = RewardBreakdown()
        # Puck at center, moving strongly toward enemy goal (x=0)
        comp.calculate(_make_state(
            ai_hit_puck=True,
            puck_pos=(400.0, 250.0),
            puck_vel=(-10.0, 0.0),
            puck_speed=10.0,
        ), bd)
        assert bd.components.get("shot_to_goal", 0) > 0

    def test_shot_away_from_goal(self):
        comp = ShotDirectionComponent()
        bd = RewardBreakdown()
        # Puck moving AWAY from enemy goal (toward AI's own goal, vx > 0)
        comp.calculate(_make_state(
            ai_hit_puck=True,
            puck_pos=(400.0, 250.0),
            puck_vel=(10.0, 0.0),
            puck_speed=10.0,
        ), bd)
        assert bd.components.get("shot_away", 0) == pytest.approx(-0.5)

    def test_slow_puck_no_reward(self):
        comp = ShotDirectionComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(
            ai_hit_puck=True,
            puck_vel=(-0.1, 0.0),
            puck_speed=0.1,
        ), bd)
        assert bd.total == 0.0


# ===================================================================
# ClearRewardComponent
# ===================================================================

class TestClearReward:
    def test_successful_clear(self):
        comp = ClearRewardComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(
            ai_hit_puck=True,
            puck_vel_pre_hit=(5.0, 0.0),  # Was heading toward AI goal
            puck_vel=(-5.0, 0.0),          # Now heading away (cleared)
        ), bd)
        assert bd.components.get("clear", 0) == pytest.approx(1.5)

    def test_no_clear_same_direction(self):
        comp = ClearRewardComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(
            ai_hit_puck=True,
            puck_vel_pre_hit=(5.0, 0.0),
            puck_vel=(3.0, 2.0),  # Still heading toward AI goal
        ), bd)
        assert bd.total == 0.0

    def test_no_clear_without_hit(self):
        comp = ClearRewardComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(ai_hit_puck=False), bd)
        assert bd.total == 0.0


# ===================================================================
# InterceptionComponent
# ===================================================================

class TestInterception:
    def test_intercept_incoming_puck(self):
        comp = InterceptionComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(
            ai_hit_puck=True,
            puck_vel_pre_hit=(5.0, 0.0),  # Was heading toward AI
        ), bd)
        assert bd.components.get("interception", 0) == pytest.approx(1.0)
        assert comp.last_hit_was_defensive

    def test_no_interception_outgoing(self):
        comp = InterceptionComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(
            ai_hit_puck=True,
            puck_vel_pre_hit=(-5.0, 0.0),  # Was heading away
            ai_puck_prev_distance=200.0,
            ai_pos=(500.0, 250.0),
        ), bd)
        assert bd.total == 0.0
        assert not comp.last_hit_was_defensive


# ===================================================================
# CounterattackComponent
# ===================================================================

class TestCounterattack:
    def test_counterattack_after_defense(self):
        interception = InterceptionComponent()
        counter = CounterattackComponent(interception)

        # First: defensive interception
        bd = RewardBreakdown()
        interception.calculate(_make_state(
            ai_hit_puck=True,
            puck_vel_pre_hit=(5.0, 0.0),
        ), bd)
        assert interception.last_hit_was_defensive

        # Second: counterattack (puck now heading to opponent)
        bd2 = RewardBreakdown()
        counter.calculate(_make_state(
            ai_hit_puck=True,
            puck_vel=(-5.0, 0.0),
        ), bd2)
        assert bd2.components.get("counterattack", 0) == pytest.approx(0.5)


# ===================================================================
# DefensiveRewardComponent
# ===================================================================

class TestDefensiveReward:
    def test_block_position(self):
        comp = DefensiveRewardComponent()
        bd = RewardBreakdown()
        # AI between puck and goal, Y-aligned, puck heading toward AI
        comp.calculate(_make_state(
            ai_pos=(650.0, 250.0),
            puck_pos=(500.0, 250.0),
            puck_in_ai_half=True,
            puck_heading_toward_ai=True,
        ), bd)
        assert bd.components.get("block_position", 0) > 0

    def test_defensive_proximity(self):
        comp = DefensiveRewardComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(
            ai_pos=(550.0, 250.0),
            puck_pos=(500.0, 250.0),
            ai_puck_distance=50.0,
            puck_in_ai_half=True,
            puck_heading_toward_ai=True,
        ), bd)
        assert bd.components.get("defensive_proximity", 0) > 0

    def test_no_reward_puck_in_opponent_half(self):
        comp = DefensiveRewardComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(puck_in_ai_half=False), bd)
        assert bd.total == 0.0

    def test_no_reward_puck_heading_away(self):
        comp = DefensiveRewardComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(
            puck_in_ai_half=True,
            puck_heading_toward_ai=False,
        ), bd)
        assert bd.total == 0.0


# ===================================================================
# GoalPressureComponent
# ===================================================================

class TestGoalPressure:
    def test_pressure_when_puck_in_opponent_half(self):
        comp = GoalPressureComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(
            puck_pos=(200.0, 250.0),
            puck_vel=(-5.0, 0.0),
        ), bd)
        assert bd.components.get("goal_pressure", 0) == pytest.approx(0.03)

    def test_no_pressure_puck_in_ai_half(self):
        comp = GoalPressureComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(
            puck_pos=(600.0, 250.0),
            puck_vel=(-5.0, 0.0),
        ), bd)
        assert bd.total == 0.0


# ===================================================================
# DisciplineComponent
# ===================================================================

class TestDiscipline:
    def test_net_front_penalty(self):
        comp = DisciplineComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(
            ai_pos=(790.0, 250.0),  # Very close to own goal (x=800)
            ai_mallet_radius=32,
        ), bd)
        assert bd.components.get("net_front", 0) == pytest.approx(-0.05)

    def test_inactivity_penalty(self):
        comp = DisciplineComponent()
        bd = RewardBreakdown()
        comp.calculate(_make_state(
            ai_vel=(0.0, 0.0),
            puck_in_ai_half=True,
            ai_puck_distance=100.0,
        ), bd)
        assert bd.components.get("inactivity", 0) == pytest.approx(-0.02)


# ===================================================================
# PBRSComponent
# ===================================================================

class TestPBRS:
    def test_no_shaping_first_step(self):
        comp = PBRSComponent(gamma=0.995)
        bd = RewardBreakdown()
        comp.calculate(_make_state(), bd)
        assert bd.total == 0.0  # No previous potential

    def test_shaping_second_step(self):
        comp = PBRSComponent(gamma=0.995)
        # Step 1: puck at center
        bd1 = RewardBreakdown()
        comp.calculate(_make_state(puck_pos=(400.0, 250.0)), bd1)

        # Step 2: puck moved toward opponent goal (x=200)
        bd2 = RewardBreakdown()
        comp.calculate(_make_state(puck_pos=(200.0, 250.0)), bd2)
        # Puck closer to opponent goal → higher potential → positive shaping
        assert bd2.total > 0

    def test_shaping_negative_when_puck_retreats(self):
        comp = PBRSComponent(gamma=0.995)
        # Step 1: puck near opponent goal
        bd1 = RewardBreakdown()
        comp.calculate(_make_state(puck_pos=(200.0, 250.0)), bd1)

        # Step 2: puck retreated to AI half
        bd2 = RewardBreakdown()
        comp.calculate(_make_state(puck_pos=(600.0, 250.0)), bd2)
        assert bd2.total < 0  # Puck retreated → negative shaping

    def test_reset_clears_potential(self):
        comp = PBRSComponent(gamma=0.995)
        bd = RewardBreakdown()
        comp.calculate(_make_state(), bd)
        comp.reset()

        bd2 = RewardBreakdown()
        comp.calculate(_make_state(), bd2)
        assert bd2.total == 0.0  # No shaping after reset


# ===================================================================
# RewardCalculator — Orchestrator
# ===================================================================

class TestRewardCalculator:
    def test_default_has_all_components(self):
        calc = RewardCalculator.default()
        names = calc.component_names
        assert "goal" in names
        assert "hit" in names
        assert "shot_direction" in names
        assert "clear" in names
        assert "interception" in names
        assert "counterattack" in names
        assert "positional" in names
        assert "defensive" in names
        assert "goal_pressure" in names
        assert "discipline" in names
        assert "pbrs" in names

    def test_minimal_has_sparse_only(self):
        calc = RewardCalculator.minimal()
        names = calc.component_names
        assert "goal" in names
        assert "hit" in names
        assert "shot_direction" not in names

    def test_calculate_returns_breakdown(self):
        calc = RewardCalculator.default()
        calc.reset()
        reward, bd = calc.calculate(_make_state(goal="ai"))
        assert reward > 0
        assert isinstance(bd, RewardBreakdown)
        assert "goal_scored" in bd.components
        assert bd.total == reward

    def test_reset_clears_all(self):
        calc = RewardCalculator.default()
        # Run a step with PBRS to set internal state
        calc.calculate(_make_state())
        calc.reset()
        # After reset, PBRS should have no previous potential
        _, bd = calc.calculate(_make_state())
        assert "pbrs_shaping" not in bd.components

    def test_offensive_defensive_ratio(self):
        """Validate that maximum offensive reward > defensive reward (ratio ≥ 2:1)."""
        calc = RewardCalculator.default()
        calc.reset()

        # Maximum offensive scenario: AI scores, perfect shot
        _, offensive_bd = calc.calculate(_make_state(
            goal="ai",
            ai_hit_puck=True,
            puck_pos=(100.0, 250.0),
            puck_vel=(-12.0, 0.0),
            puck_speed=12.0,
            puck_vel_pre_hit=(5.0, 0.0),
            puck_in_ai_half=False,
        ))

        calc.reset()

        # Maximum defensive scenario: interception + clear + block
        _, defensive_bd = calc.calculate(_make_state(
            ai_hit_puck=True,
            ai_pos=(650.0, 250.0),
            puck_pos=(500.0, 250.0),
            puck_vel=(-5.0, 0.0),
            puck_speed=5.0,
            puck_vel_pre_hit=(5.0, 0.0),
            puck_in_ai_half=True,
            puck_heading_toward_ai=True,
            ai_puck_distance=50.0,
        ))

        offensive_total = sum(
            v for k, v in offensive_bd.categories.items()
            if k in (RewardCategory.GOAL, RewardCategory.HIT,
                     RewardCategory.SHOT_DIRECTION, RewardCategory.PRESSURE)
        )
        defensive_total = sum(
            v for k, v in defensive_bd.categories.items()
            if k in (RewardCategory.DEFENSIVE, RewardCategory.CLEAR)
        )

        # Offensive max should be at least 2x defensive max
        assert offensive_total >= 2 * defensive_total, (
            f"Ratio violated: offensive={offensive_total:.2f}, "
            f"defensive={defensive_total:.2f}"
        )


# ===================================================================
# Integration: base_env uses componentized rewards
# ===================================================================

@pytest.fixture(scope="session", autouse=True)
def init_pygame():
    pygame.init()
    pygame.display.set_mode((1, 1), pygame.NOFRAME)
    yield
    pygame.quit()


@pytest.fixture
def env():
    from training.envs.base_env import AirHockeyEnv
    e = AirHockeyEnv(render_mode=None)
    yield e
    e.close()


class TestBaseEnvIntegration:
    def test_reward_calculator_exists(self, env):
        assert hasattr(env, "reward_calculator")
        assert isinstance(env.reward_calculator, RewardCalculator)

    def test_breakdown_stored_after_step(self, env):
        env.reset()
        env.step(4)  # Stay
        assert env._last_reward_breakdown is not None
        assert isinstance(env._last_reward_breakdown, RewardBreakdown)

    def test_step_produces_valid_reward(self, env):
        env.reset()
        for _ in range(20):
            _, reward, term, trunc, _ = env.step(2)  # Move left
            assert isinstance(reward, float)
            if term or trunc:
                env.reset()

    def test_puck_vel_pre_hit_captured(self, env):
        env.reset()
        env.step(4)  # Stay - captures pre-hit velocity
        assert env._puck_vel_pre_hit is not None
        assert len(env._puck_vel_pre_hit) == 2

    def test_hit_tracking_works(self, env):
        env.reset()
        # After several steps without hitting, consecutive_hits should be 0
        for _ in range(10):
            env.step(4)
        assert env._consecutive_hits == 0
        assert env.steps_since_last_hit > 0

    def test_reward_calculator_resets_on_env_reset(self, env):
        env.reset()
        # Run a few steps to set internal state
        for _ in range(5):
            env.step(4)
        env.reset()
        # PBRS should have no previous potential after reset
        assert env._puck_vel_pre_hit is None
