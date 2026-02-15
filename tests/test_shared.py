"""
Tests for shared configuration, physics utilities and base entities.
Validates that GameConfig, PhysicsConfig, Puck, Mallet, and Table
behave consistently — these are the foundations of both game and training.
"""
import math
import pytest
import pygame
import numpy as np

from shared.config import GameConfig, PhysicsConfig, COLORS, TRAINING_WIDTH, TRAINING_HEIGHT
from shared.physics import (
    calculate_vector, vector_length, normalize_vector,
    dot_product, line_circle_intersection,
)
from shared.entities.puck import Puck
from shared.entities.mallet import Mallet
from shared.entities.table import Table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def init_pygame():
    """Initialize pygame once for the whole test session."""
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


@pytest.fixture
def puck(config, physics):
    return Puck(config, physics)


@pytest.fixture
def mallet(config):
    return Mallet(200.0, 250.0, config=config)


@pytest.fixture
def table(config, physics):
    return Table(config, physics)


# ===================================================================
# GameConfig
# ===================================================================

class TestGameConfig:
    def test_defaults(self, config):
        assert config.width == 800
        assert config.height == 500
        assert config.fps == 120

    def test_derived_fields(self, config):
        assert config.half_width == 400
        assert config.scale_factor == pytest.approx(1.0)
        assert config.puck_radius == 15
        assert config.mallet_radius == 32

    def test_custom_resolution_scaling(self):
        cfg = GameConfig(width=1200, height=750)
        assert cfg.scale_factor == pytest.approx(1.5)
        assert cfg.puck_radius == int(15 * 1.5)
        assert cfg.mallet_radius == int(32 * 1.5)

    def test_small_resolution_clamps(self):
        cfg = GameConfig(width=200, height=125)
        assert cfg.puck_radius >= 8
        assert cfg.mallet_radius >= 16

    def test_training_dimensions_match_defaults(self):
        """Training must use same dims as default config."""
        assert TRAINING_WIDTH == 800
        assert TRAINING_HEIGHT == 500


# ===================================================================
# PhysicsConfig
# ===================================================================

class TestPhysicsConfig:
    def test_defaults(self, physics):
        assert 0.99 <= physics.friction <= 1.0
        assert physics.max_puck_speed > 0
        assert 0 < physics.collision_elasticity <= 1.0
        assert 0 < physics.goal_width_ratio < 1

    def test_ai_move_amount_matches_training(self, physics):
        """ai_move_amount must match training env step size."""
        assert physics.ai_move_amount == 5


# ===================================================================
# Physics helpers
# ===================================================================

class TestPhysics:
    def test_calculate_vector_lists(self):
        v = calculate_vector([0, 0], [3, 4])
        assert v == [3, 4]

    def test_vector_length(self):
        assert vector_length([3, 4]) == pytest.approx(5.0)
        assert vector_length([0, 0]) == 0.0

    def test_normalize(self):
        n = normalize_vector([3, 4])
        assert vector_length(n) == pytest.approx(1.0)

    def test_normalize_zero(self):
        assert normalize_vector([0, 0]) == [0.0, 0.0]

    def test_dot_product(self):
        assert dot_product([1, 0], [0, 1]) == 0.0
        assert dot_product([2, 3], [4, 5]) == pytest.approx(23.0)

    def test_line_circle_hit(self):
        result = line_circle_intersection([0, 0], [10, 0], [5, 0], 2)
        assert result is not None

    def test_line_circle_miss(self):
        result = line_circle_intersection([0, 0], [10, 0], [5, 50], 2)
        assert result is None


# ===================================================================
# Puck
# ===================================================================

class TestPuck:
    def test_initial_position(self, puck, config):
        assert puck.position == [float(config.width // 2), float(config.height // 2)]

    def test_has_velocity(self, puck):
        assert len(puck.velocity) == 2

    def test_update_moves_puck(self, puck):
        old = puck.position.copy()
        puck.update()
        # Puck should have moved (it has initial velocity)
        assert puck.position != old or puck.velocity == [0.0, 0.0]

    def test_stays_in_bounds(self, puck, config):
        """Puck must never leave the field after updates."""
        puck.velocity = [50, 50]
        for _ in range(200):
            puck.update()
        assert puck.radius <= puck.position[0] <= config.width - puck.radius
        assert puck.radius <= puck.position[1] <= config.height - puck.radius

    def test_speed_clamped(self, puck):
        puck.velocity = [100, 100]
        puck.update()
        speed = vector_length(puck.velocity)
        assert speed <= puck.max_speed + 2  # small tolerance for corner force

    def test_reset_center(self, puck, config):
        puck.position = [10.0, 10.0]
        puck.reset()
        assert puck.position == [float(config.width // 2), float(config.height // 2)]

    def test_reset_scorer_direction(self, puck):
        puck.reset(scorer="player")
        assert puck.velocity[0] < 0  # toward AI

        puck.reset(scorer="ai")
        assert puck.velocity[0] > 0  # toward player

    def test_rect_synced(self, puck):
        puck.update()
        assert puck.rect.center == (int(puck.position[0]), int(puck.position[1]))


# ===================================================================
# Mallet
# ===================================================================

class TestMallet:
    def test_initial_position(self, mallet):
        assert mallet.position == [200.0, 250.0]

    def test_velocity_init(self, mallet):
        assert mallet.velocity == [0.0, 0.0]

    def test_power_up_size_modifier(self, mallet):
        original = mallet.radius
        mallet.apply_size_modifier(2.0)
        assert mallet.radius == max(8, int(original * 2.0))
        mallet.apply_size_modifier(1.0)
        assert mallet.radius == original

    def test_speed_multiplier_default(self, mallet):
        assert mallet.speed_multiplier == 1.0


# ===================================================================
# Table
# ===================================================================

class TestTable:
    def test_goal_dimensions(self, table, config, physics):
        expected = config.height * physics.goal_width_ratio
        assert table.goal_width == pytest.approx(expected)

    def test_goal_y_bounds(self, table, config):
        assert table.goal_y1 < table.goal_y2
        assert table.goal_y1 >= 0
        assert table.goal_y2 <= config.height

    def test_no_goal_when_center(self, table, puck):
        """Puck at center should not register as goal."""
        result = table.is_goal(puck)
        assert result is None

    def test_draw_creates_hitboxes(self, table, config):
        screen = pygame.Surface((config.width, config.height))
        table.draw(screen)
        assert table.goal_left_hitbox is not None
        assert table.goal_right_hitbox is not None

    def test_goal_detection_left(self, table, config, puck):
        """Puck inside left goal hitbox should register as AI goal."""
        screen = pygame.Surface((config.width, config.height))
        table.draw(screen)
        # Place puck inside left goal
        hb = table.goal_left_hitbox
        puck.position = [float(hb.centerx), float(hb.centery)]
        puck.rect.center = (hb.centerx, hb.centery)
        assert table.is_goal(puck) == "ai"

    def test_goal_detection_right(self, table, config, puck):
        """Puck inside right goal hitbox should register as player goal."""
        screen = pygame.Surface((config.width, config.height))
        table.draw(screen)
        hb = table.goal_right_hitbox
        puck.position = [float(hb.centerx), float(hb.centery)]
        puck.rect.center = (hb.centerx, hb.centery)
        assert table.is_goal(puck) == "player"
