"""
Integration tests — verify cross-module consistency and end-to-end flows.
These catch mismatches between game and training that unit tests might miss.
"""
import math
import pytest
import pygame
import numpy as np

from shared.config import GameConfig, PhysicsConfig, TRAINING_WIDTH, TRAINING_HEIGHT
from shared.entities.puck import Puck
from shared.entities.mallet import Mallet
from shared.entities.table import Table


@pytest.fixture(scope="session", autouse=True)
def init_pygame():
    pygame.init()
    pygame.display.set_mode((1, 1), pygame.NOFRAME)
    yield
    pygame.quit()


@pytest.fixture
def config():
    return GameConfig()


@pytest.fixture
def physics():
    return PhysicsConfig()


# ===================================================================
# Observation ↔ Environment alignment
# ===================================================================

class TestObsEnvAlignment:
    """Ensure observation builder produces vectors identical to training env."""

    def test_obs_dimensions_match_env(self):
        """Game obs builder v2 must match training env obs space."""
        from training.envs.base_env import AirHockeyEnv
        from game.ai.observation_builder import create_observation

        env = AirHockeyEnv(render_mode=None)
        obs_env, _ = env.reset()

        cfg = GameConfig()
        phys = PhysicsConfig()
        ai = Mallet(float(cfg.width * 3 // 4), float(cfg.height // 2), config=cfg)
        puck = Puck(cfg, phys)
        human = Mallet(float(cfg.width // 4), float(cfg.height // 2), config=cfg)

        obs_game = create_observation(ai, puck, human, 0, 0, model_type="v2")

        assert obs_game.shape == obs_env.shape, \
            f"Shape mismatch: game {obs_game.shape} vs env {obs_env.shape}"
        assert obs_game.dtype == obs_env.dtype
        env.close()

    def test_obs_powerups_dimensions_match_env(self):
        """Game obs builder v2_powerups must match powerups env obs space."""
        from training.envs.powerups_env import AirHockeyWithPowerUpsEnv
        from game.ai.observation_builder import create_observation

        env = AirHockeyWithPowerUpsEnv(render_mode=None)
        obs_env, _ = env.reset()

        cfg = GameConfig()
        phys = PhysicsConfig()
        ai = Mallet(float(cfg.width * 3 // 4), float(cfg.height // 2), config=cfg)
        puck = Puck(cfg, phys)
        human = Mallet(float(cfg.width // 4), float(cfg.height // 2), config=cfg)

        obs_game = create_observation(ai, puck, human, 0, 0, model_type="v2_powerups")
        assert obs_game.shape == obs_env.shape
        env.close()


# ===================================================================
# Puck ↔ Table goal detection
# ===================================================================

class TestGoalDetection:
    """Full cycle: puck movement → goal detection → score update."""

    def test_fast_puck_scores_goal(self, config, physics):
        """A puck placed inside the goal area should trigger detection."""
        table = Table(config, physics)
        screen = pygame.Surface((config.width, config.height))
        table.draw(screen)  # initialize hitboxes

        puck = Puck(config, physics)
        # Place puck directly inside right goal hitbox
        hb = table.goal_right_hitbox
        puck.position = [float(hb.centerx), float(hb.centery)]
        puck.rect.center = (hb.centerx, hb.centery)
        puck.velocity = [5.0, 0.0]

        result = table.is_goal(puck)
        assert result == "player", "Puck inside right goal should register"

    def test_puck_bounces_off_goal_frame(self, config, physics):
        """Puck hitting the goal frame (above/below opening) should bounce, not score."""
        table = Table(config, physics)
        screen = pygame.Surface((config.width, config.height))
        table.draw(screen)

        puck = Puck(config, physics)
        # Aim above the goal opening
        puck.position = [float(config.width - 50), float(table.goal_y1 - 30)]
        puck.velocity = [8.0, 0.0]

        scored = False
        for _ in range(30):
            puck.update()
            result = table.is_goal(puck)
            if result is not None:
                scored = True
                break

        assert not scored, "Puck above goal should not score"


# ===================================================================
# Physics consistency across 200 frames
# ===================================================================

class TestPhysicsStability:
    """Run a mini simulation to ensure no NaN/Inf or unbounded values."""

    def test_simulation_200_frames(self, config, physics):
        table = Table(config, physics)
        screen = pygame.Surface((config.width, config.height))
        table.draw(screen)

        puck = Puck(config, physics)
        ai = Mallet(600.0, 250.0, config=config)
        human = Mallet(200.0, 250.0, config=config)

        puck.velocity = [8.0, -5.0]

        for frame in range(200):
            puck.update()

            # No NaN/Inf
            for v in puck.position + puck.velocity:
                assert math.isfinite(v), f"Frame {frame}: non-finite value in puck"

            # In bounds
            assert 0 <= puck.position[0] <= config.width
            assert 0 <= puck.position[1] <= config.height

    def test_mallet_collision_no_explode(self, config, physics):
        """Repeated mallet-puck collisions should not produce unbounded speed."""
        puck = Puck(config, physics)
        mallet = Mallet(400.0, 250.0, config=config)

        for _ in range(50):
            # Place puck on mallet
            puck.position = [mallet.position[0] + mallet.radius + puck.radius - 1,
                             mallet.position[1]]
            puck.velocity = [-5.0, 0.0]
            mallet.velocity = [3.0, 0.0]
            puck.rect.center = (int(puck.position[0]), int(puck.position[1]))

            puck.check_mallet_collision(mallet)
            speed = math.hypot(puck.velocity[0], puck.velocity[1])
            assert speed <= puck.max_speed + 5, f"Speed exploded: {speed}"


# ===================================================================
# Training env ↔ SB3 check_env
# ===================================================================

class TestSB3Compatibility:
    """Validate env with Stable-Baselines3 check_env."""

    @pytest.mark.timeout(30)
    def test_check_env_base(self):
        from stable_baselines3.common.env_checker import check_env
        from training.envs.base_env import AirHockeyEnv
        env = AirHockeyEnv(render_mode=None)
        # check_env raises on failure
        check_env(env, warn=True, skip_render_check=True)
        env.close()

    @pytest.mark.timeout(30)
    def test_check_env_powerups(self):
        from stable_baselines3.common.env_checker import check_env
        from training.envs.powerups_env import AirHockeyWithPowerUpsEnv
        env = AirHockeyWithPowerUpsEnv(render_mode=None)
        check_env(env, warn=True, skip_render_check=True)
        env.close()


# ===================================================================
# Level configs load and produce valid mechanics
# ===================================================================

class TestLevelConfigIntegration:
    """All 5 levels should produce a valid mechanic and have correct keys."""

    def test_all_levels_load(self):
        from game.config.level_config import get_level_config
        from game.core.mechanics import create_mechanic, LevelMechanic

        cfg = GameConfig()
        for level_id in range(1, 6):
            lcfg = get_level_config(level_id)
            assert "name" in lcfg, f"Level {level_id} missing 'name'"
            mech = create_mechanic(cfg, lcfg)
            assert isinstance(mech, LevelMechanic), \
                f"Level {level_id} mechanic is not a LevelMechanic"

    def test_level_1_no_mechanic(self):
        from game.config.level_config import get_level_config
        from game.core.mechanics import create_mechanic, NoneMechanic

        cfg = GameConfig()
        lcfg = get_level_config(1)
        mech = create_mechanic(cfg, lcfg)
        assert isinstance(mech, NoneMechanic)
