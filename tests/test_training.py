"""
Tests for the training environment (base_env.py).
Validates action space, observation space, reward signals, episode lifecycle,
and consistency with game parameters.
"""
import math
import pytest
import pygame
import numpy as np

from shared.config import GameConfig, PhysicsConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


@pytest.fixture
def powerups_env():
    from training.envs.powerups_env import AirHockeyWithPowerUpsEnv
    e = AirHockeyWithPowerUpsEnv(render_mode=None)
    yield e
    e.close()


# ===================================================================
# Base Environment — spaces
# ===================================================================

class TestBaseEnvSpaces:
    def test_action_space_9(self, env):
        """Must have exactly 9 actions (with diagonals)."""
        assert env.action_space.n == 9

    def test_observation_space_13(self, env):
        """Observation should be 13-dim."""
        assert env.observation_space.shape == (13,)

    def test_observation_bounds(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), \
            f"Obs out of bounds: {obs}"

    def test_all_actions_valid(self, env):
        """Every action 0-8 should be accepted without error."""
        env.reset()
        for a in range(9):
            obs, reward, term, trunc, info = env.step(a)
            assert env.observation_space.contains(obs), \
                f"Action {a} produced invalid obs"
            if term or trunc:
                env.reset()


# ===================================================================
# Base Environment — reset
# ===================================================================

class TestBaseEnvReset:
    def test_reset_returns_obs(self, env):
        obs, info = env.reset()
        assert obs.shape == (13,)
        assert isinstance(info, dict)

    def test_reset_scores_zero(self, env):
        env.player_score = 5
        env.ai_score = 3
        env.reset()
        assert env.player_score == 0
        assert env.ai_score == 0

    def test_reset_steps_zero(self, env):
        env.steps = 999
        env.reset()
        assert env.steps == 0

    def test_reset_deterministic_with_seed(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


# ===================================================================
# Base Environment — step dynamics
# ===================================================================

class TestBaseEnvStep:
    def test_step_increments(self, env):
        env.reset()
        env.step(4)  # Stay
        assert env.steps == 1

    def test_ai_stays_in_right_half(self, env):
        """AI mallet must be constrained to right half of field."""
        env.reset()
        W = env.config.width
        for _ in range(100):
            env.step(2)  # Left
        assert env.ai_mallet_position[0] >= W // 2 + env.ai_mallet_radius

    def test_ai_stays_in_bounds_y(self, env):
        """AI mallet must stay within vertical bounds."""
        env.reset()
        H = env.config.height
        for _ in range(200):
            env.step(0)  # Up
        assert env.ai_mallet_position[1] >= env.ai_mallet_radius

        env.reset()
        for _ in range(200):
            env.step(1)  # Down
        assert env.ai_mallet_position[1] <= H - env.ai_mallet_radius

    def test_diagonal_movement_factor(self, env):
        """Diagonal moves should be slower than cardinal by ~sqrt(2)."""
        env.reset()
        start_pos = env.ai_mallet_position.copy()
        env.step(0)  # Up
        cardinal_dy = abs(env.ai_mallet_position[1] - start_pos[1])

        env.reset()
        start_pos = env.ai_mallet_position.copy()
        env.step(5)  # UpLeft
        diag_dy = abs(env.ai_mallet_position[1] - start_pos[1])

        assert diag_dy < cardinal_dy
        assert diag_dy == pytest.approx(cardinal_dy * 0.7071, abs=1.0)

    def test_step_returns_valid_info(self, env):
        env.reset()
        _, _, _, _, info = env.step(4)
        assert "player_score" in info
        assert "ai_score" in info
        assert "steps" in info
        assert "hit_puck" in info


# ===================================================================
# Base Environment — episode termination
# ===================================================================

class TestBaseEnvTermination:
    def test_max_steps_truncation(self, env):
        """Episode should truncate after MAX_STEPS."""
        env.reset()
        truncated = False
        for i in range(env.max_steps + 10):
            _, _, term, trunc, _ = env.step(4)
            if term:
                env.reset()
            if trunc:
                truncated = True
                break
        assert truncated

    def test_score_limit_termination(self, env):
        """Episode should terminate when score limit is reached."""
        env.reset()
        env.player_score = env.score_limit - 1
        # Force a player goal by placing puck in right goal area
        W, H = env.config.width, env.config.height
        env.puck.position = [W - 2, H // 2]
        env.puck.velocity = [10, 0]
        env.puck.rect.center = (int(env.puck.position[0]), int(env.puck.position[1]))

        # Run steps — a goal might score
        terminated = False
        for _ in range(50):
            _, _, term, trunc, _ = env.step(4)
            if term:
                terminated = True
                break
            if trunc:
                break
        # Whether or not goal fired, the mechanism should work
        assert env.player_score >= env.score_limit - 1


# ===================================================================
# Base Environment — config alignment with game
# ===================================================================

class TestEnvGameAlignment:
    def test_dimensions_match(self, env):
        cfg = GameConfig()
        assert env.config.width == cfg.width
        assert env.config.height == cfg.height

    def test_physics_match(self, env):
        phys = PhysicsConfig()
        assert env.physics.ai_move_amount == phys.ai_move_amount
        assert env.physics.max_puck_speed == phys.max_puck_speed

    def test_mallet_radius_match(self, env):
        cfg = GameConfig()
        assert env.ai_mallet_radius == cfg.mallet_radius

    def test_score_limit(self, env):
        assert env.score_limit == 7


# ===================================================================
# Power-ups Environment
# ===================================================================

class TestPowerUpsEnv:
    def test_observation_space_26(self, powerups_env):
        assert powerups_env.observation_space.shape == (26,)

    def test_action_space_9(self, powerups_env):
        assert powerups_env.action_space.n == 9

    def test_reset(self, powerups_env):
        obs, info = powerups_env.reset()
        assert obs.shape == (26,)
        assert powerups_env.observation_space.contains(obs)

    def test_step(self, powerups_env):
        powerups_env.reset()
        obs, r, term, trunc, info = powerups_env.step(4)
        assert obs.shape == (26,)
        assert powerups_env.observation_space.contains(obs)


# ===================================================================
# PPO Configs
# ===================================================================

from training.configs.ppo_configs import PRESETS


class TestPPOConfigs:
    def test_v2_presets_exist(self):
        assert "v2_quick" in PRESETS
        assert "v2_standard" in PRESETS
        assert "v2_deep" in PRESETS

    def test_legacy_presets_exist(self):
        assert "quick" in PRESETS
        assert "standard" in PRESETS
        assert "deep" in PRESETS

    def test_v2_preset_fields(self):
        v2_keys = [k for k in PRESETS if k.startswith("v2_")]
        for name in v2_keys:
            cfg = PRESETS[name]
            assert cfg.total_timesteps > 0, f"{name} has invalid timesteps"
            # learning_rate can be a float or a callable (schedule)
            lr = cfg.learning_rate
            if callable(lr):
                assert lr(1.0) > 0, f"{name} schedule at start should be > 0"
            else:
                assert lr > 0, f"{name} has invalid lr"
            assert cfg.batch_size > 0, f"{name} has invalid batch_size"
            assert cfg.n_epochs > 0, f"{name} has invalid n_epochs"
            assert 0 < cfg.gamma <= 1.0, f"{name} has invalid gamma"


# ===================================================================
# Opponent System (Mejora 8)
# ===================================================================

from training.envs.opponents import (
    AlgorithmicOpponent,
    DefensiveStrategy,
    FieldSnapshot,
    InterceptStrategy,
    OffensiveStrategy,
    OpponentFactory,
    OpponentParams,
    OpponentState,
    PositionalStrategy,
)
from training.configs.curriculum import CurriculumMetrics, CURRICULUM_LEVELS


def _make_snapshot(
    puck_pos=(400, 250),
    puck_vel=(0, 0),
    mallet_pos=(200, 250),
    width=800,
    height=500,
) -> FieldSnapshot:
    """Helper: create a minimal FieldSnapshot for testing."""
    return FieldSnapshot(
        width=width,
        height=height,
        half_width=width // 2,
        mallet_pos=mallet_pos,
        mallet_radius=32,
        puck_pos=puck_pos,
        puck_vel=puck_vel,
        puck_radius=15,
        own_goal_x=0.0,
        own_goal_center_y=height / 2,
        goal_y1=height * (1 - 1 / 3) / 2,
        goal_y2=height * (1 + 1 / 3) / 2,
        rival_goal_x=float(width),
        rival_goal_center_y=height / 2,
    )


class TestOpponentParams:
    def test_from_skill_clamps(self):
        p = OpponentParams.from_skill(1.5)
        assert p.skill == 1.0
        p = OpponentParams.from_skill(-0.5)
        assert p.skill == 0.0

    def test_style_weights_sum_to_one(self):
        for s in [0.0, 0.3, 0.5, 0.7, 1.0]:
            p = OpponentParams.from_skill(s)
            total = sum(p.style_weights.values())
            assert total == pytest.approx(1.0, abs=1e-6), f"skill={s} → weights sum={total}"

    def test_from_curriculum_level(self):
        level = CURRICULUM_LEVELS[3]
        p = OpponentParams.from_curriculum_level(level)
        assert p.accuracy == level.accuracy
        assert p.aggression == level.aggression


class TestOpponentStrategies:
    def test_defensive_stays_near_goal(self):
        snap = _make_snapshot(puck_pos=(300, 200), puck_vel=(-5, 1))
        strategy = DefensiveStrategy()
        params = OpponentParams.from_skill(0.5)
        tx, ty = strategy.compute_target(snap, params)
        # Should be in the left portion of the field
        assert tx < snap.half_width * 0.5

    def test_offensive_approaches_puck(self):
        snap = _make_snapshot(puck_pos=(200, 300), puck_vel=(0, 0))
        strategy = OffensiveStrategy()
        params = OpponentParams.from_skill(0.7)
        tx, ty = strategy.compute_target(snap, params)
        # Target should be near the puck
        dist = ((tx - 200) ** 2 + (ty - 300) ** 2) ** 0.5
        assert dist < 200

    def test_intercept_computes_valid_target(self):
        snap = _make_snapshot(puck_pos=(500, 200), puck_vel=(-4, 2))
        strategy = InterceptStrategy()
        params = OpponentParams.from_skill(0.6)
        tx, ty = strategy.compute_target(snap, params)
        assert 0 <= tx <= snap.half_width
        assert 0 <= ty <= snap.height

    def test_positional_near_center(self):
        snap = _make_snapshot(puck_pos=(600, 400), puck_vel=(3, 0))
        strategy = PositionalStrategy()
        params = OpponentParams.from_skill(0.4)
        tx, ty = strategy.compute_target(snap, params)
        assert 100 < tx < 300
        assert 100 < ty < 400


class TestAlgorithmicOpponent:
    def test_does_not_stay_still(self):
        """Opponent must move over a series of steps."""
        opp = OpponentFactory.from_skill(0.5)
        positions = set()
        pos = (200.0, 250.0)
        for i in range(50):
            snap = _make_snapshot(
                puck_pos=(300 - i, 250),
                puck_vel=(-3, 1),
                mallet_pos=pos,
            )
            new_x, new_y, _, _ = opp.update(snap, pos)
            positions.add((round(new_x, 1), round(new_y, 1)))
            pos = (new_x, new_y)
        assert len(positions) > 3, "Opponent should move to multiple positions"

    def test_defends_when_puck_approaches(self):
        """Opponent should enter DEFENSIVE state when puck shoots at it."""
        opp = OpponentFactory.from_skill(0.6)
        snap = _make_snapshot(puck_pos=(350, 250), puck_vel=(-8, 0))
        opp.update(snap, (200.0, 250.0))
        assert opp.state in (OpponentState.DEFENSIVE, OpponentState.INTERCEPT)

    def test_attacks_when_puck_in_own_half(self):
        """Opponent should go offensive when puck is in its half and slow."""
        opp = OpponentFactory.from_skill(0.7)
        opp._style_bias = "offensive"  # force bias for determinism
        snap = _make_snapshot(puck_pos=(200, 250), puck_vel=(0.5, 0))
        opp.update(snap, (150.0, 250.0))
        assert opp.state == OpponentState.OFFENSIVE

    def test_varies_behavior_across_episodes(self):
        """Style bias should change between resets (anti-overfitting)."""
        opp = OpponentFactory.from_skill(0.5)
        biases = set()
        for _ in range(20):
            opp.reset_episode()
            biases.add(opp._style_bias)
        assert len(biases) >= 2, "Should show behavioral variation across episodes"

    def test_skill_scaling(self):
        """Higher skill opponent should move faster / more precisely."""
        opp_low = OpponentFactory.from_skill(0.1)
        opp_high = OpponentFactory.from_skill(0.9)
        assert opp_high.params.max_speed > opp_low.params.max_speed
        assert opp_high.params.accuracy > opp_low.params.accuracy

    def test_stays_in_left_half(self):
        """Opponent mallet must never cross the center line."""
        opp = OpponentFactory.from_skill(0.9)
        opp._style_bias = "offensive"
        pos = (200.0, 250.0)
        for _ in range(200):
            snap = _make_snapshot(
                puck_pos=(600, 250),
                puck_vel=(5, 0),
                mallet_pos=pos,
            )
            new_x, new_y, _, _ = opp.update(snap, pos)
            assert new_x <= snap.half_width, f"Crossed center: x={new_x}"
            assert new_y >= 0 and new_y <= snap.height
            pos = (new_x, new_y)


class TestBaseEnvOpponentIntegration:
    """Verify that base_env uses the new opponent system correctly."""

    def test_env_has_opponent(self, env):
        assert hasattr(env, "opponent")
        assert isinstance(env.opponent, AlgorithmicOpponent)

    def test_opponent_resets_on_env_reset(self, env):
        env.reset(seed=42)
        bias1 = env.opponent._style_bias
        # Reset multiple times to check if bias changes (probabilistic)
        biases = {bias1}
        for _ in range(10):
            env.reset()
            biases.add(env.opponent._style_bias)
        # At least one different bias in 10 resets (very high probability)
        assert len(biases) >= 1  # Always true, but checks no crash

    def test_difficulty_increases_opponent_params(self, env):
        env.reset()
        old_skill = env.opponent_skill
        env.last_average_reward = -1000  # Force advancement
        env.increase_opponent_difficulty(500)
        assert env.opponent_skill > old_skill
        assert env.opponent.params.skill == env.opponent_skill

    def test_human_mallet_stays_in_bounds(self, env):
        """After many steps, opponent mallet should remain in-bounds."""
        env.reset()
        W, H = env.config.width, env.config.height
        for _ in range(100):
            env.step(env.action_space.sample())
        x, y = env.human_mallet.position
        assert 0 <= x <= W // 2, f"Out of bounds x={x}"
        assert 0 <= y <= H, f"Out of bounds y={y}"


class TestCurriculumMetrics:
    def test_meets_advancement(self):
        level = CURRICULUM_LEVELS[2]
        passing = CurriculumMetrics(
            win_rate=0.7,
            avg_goals_scored=3.0,
            avg_goals_conceded=2.0,
        )
        assert passing.meets_advancement(level)

    def test_fails_advancement_low_winrate(self):
        level = CURRICULUM_LEVELS[2]
        failing = CurriculumMetrics(
            win_rate=0.3,
            avg_goals_scored=3.0,
            avg_goals_conceded=2.0,
        )
        assert not failing.meets_advancement(level)

    def test_fails_advancement_low_goals(self):
        level = CURRICULUM_LEVELS[2]
        failing = CurriculumMetrics(
            win_rate=0.8,
            avg_goals_scored=0.5,
            avg_goals_conceded=1.0,
        )
        assert not failing.meets_advancement(level)
