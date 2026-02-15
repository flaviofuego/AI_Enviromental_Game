"""
Tests for game state machine, match manager, observation builder,
model loader utilities, and level mechanics.
"""
import math
import os
import json
import tempfile
import pytest
import pygame
import numpy as np

from shared.config import GameConfig, PhysicsConfig, COLORS, TRAINING_WIDTH, TRAINING_HEIGHT
from shared.entities.puck import Puck
from shared.entities.mallet import Mallet


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
def config():
    return GameConfig()


@pytest.fixture
def physics():
    return PhysicsConfig()


def _make_entity_trio(config, physics):
    """Create (ai_mallet, puck, human_mallet) for obs builder tests."""
    puck = Puck(config, physics)
    ai = Mallet(600.0, 250.0, config=config)
    human = Mallet(200.0, 250.0, config=config)
    return ai, puck, human


# ===================================================================
# GameState
# ===================================================================

from game.core.game_state import GamePhase, GameState


class TestGameState:
    def test_initial_phase(self):
        gs = GameState()
        assert gs.phase == GamePhase.MENU

    def test_reset_match(self):
        gs = GameState()
        gs.player_score = 5
        gs.ai_score = 3
        gs.reset_match()
        assert gs.player_score == 0
        assert gs.ai_score == 0
        assert gs.phase == GamePhase.PLAYING

    def test_record_goal_player(self):
        gs = GameState()
        gs.phase = GamePhase.PLAYING
        over = gs.record_goal("player", 7)
        assert gs.player_score == 1
        assert not over

    def test_record_goal_ai(self):
        gs = GameState()
        gs.phase = GamePhase.PLAYING
        over = gs.record_goal("ai", 7)
        assert gs.ai_score == 1
        assert not over

    def test_game_over_player_wins(self):
        gs = GameState()
        gs.player_score = 6
        over = gs.record_goal("player", 7)
        assert over
        assert gs.winner == "player"
        assert gs.phase == GamePhase.GAME_OVER

    def test_game_over_ai_wins(self):
        gs = GameState()
        gs.ai_score = 6
        over = gs.record_goal("ai", 7)
        assert over
        assert gs.winner == "ai"
        assert gs.phase == GamePhase.GAME_OVER

    def test_time_limit_no_limit(self):
        gs = GameState()
        assert gs.check_time_limit(100.0, None) is False
        assert gs.check_time_limit(100.0, 0) is False

    def test_time_limit_reached_different_scores(self):
        gs = GameState()
        gs.player_score = 3
        gs.ai_score = 1
        over = gs.check_time_limit(120.0, 60.0)
        assert over
        assert gs.winner == "player"

    def test_time_limit_tie_overtime(self):
        gs = GameState()
        gs.player_score = 2
        gs.ai_score = 2
        over = gs.check_time_limit(120.0, 60.0, overtime_on_tie=True)
        # Should NOT end because overtime_on_tie keeps playing
        assert not over


# ===================================================================
# MatchManager
# ===================================================================

from game.core.match_manager import GameMode, MatchConfig, MatchManager


class TestMatchManager:
    def test_defaults(self):
        mm = MatchManager()
        assert mm.mode == GameMode.PLAYER_VS_AI
        assert mm.score_limit == 7

    def test_custom_config(self):
        cfg = MatchConfig(
            mode=GameMode.PLAYER_VS_PLAYER,
            score_limit=5,
            powerups_enabled=True,
            level_id=3,
        )
        mm = MatchManager(cfg)
        assert mm.mode == GameMode.PLAYER_VS_PLAYER
        assert mm.score_limit == 5
        assert mm.powerups_enabled is True

    def test_lifecycle(self):
        mm = MatchManager()
        assert not mm.is_active
        mm.start_match()
        assert mm.is_active
        mm.end_match()
        assert not mm.is_active


# ===================================================================
# ObservationBuilder
# ===================================================================

from game.ai.observation_builder import create_observation


class TestObservationBuilder:
    def test_original_shape(self, config, physics):
        ai, puck, human = _make_entity_trio(config, physics)
        obs = create_observation(ai, puck, human, 0, 0, model_type="original")
        assert obs.shape == (13,)
        assert obs.dtype == np.float32

    def test_v2_shape(self, config, physics):
        ai, puck, human = _make_entity_trio(config, physics)
        obs = create_observation(ai, puck, human, 0, 0, model_type="v2")
        assert obs.shape == (13,)
        assert obs.dtype == np.float32

    def test_enhanced_shape(self, config, physics):
        ai, puck, human = _make_entity_trio(config, physics)
        obs = create_observation(ai, puck, human, 0, 0, model_type="enhanced")
        assert obs.shape == (21,)

    def test_v2_powerups_shape(self, config, physics):
        ai, puck, human = _make_entity_trio(config, physics)
        obs = create_observation(ai, puck, human, 0, 0, model_type="v2_powerups")
        assert obs.shape == (26,)

    def test_values_normalized(self, config, physics):
        """All obs values should be in [-1, 1]."""
        ai, puck, human = _make_entity_trio(config, physics)
        for mt in ("original", "v2", "enhanced", "v2_powerups"):
            obs = create_observation(ai, puck, human, 3, 2, model_type=mt)
            assert np.all(obs >= -1.0), f"{mt}: obs has values < -1"
            assert np.all(obs <= 1.0), f"{mt}: obs has values > 1"

    def test_score_encoding(self, config, physics):
        ai, puck, human = _make_entity_trio(config, physics)
        obs_0_0 = create_observation(ai, puck, human, 0, 0, model_type="v2")
        obs_3_5 = create_observation(ai, puck, human, 3, 5, model_type="v2")
        # Last two elements are player_score/7 and ai_score/7
        assert obs_0_0[-2] == pytest.approx(0.0)
        assert obs_0_0[-1] == pytest.approx(0.0)
        assert obs_3_5[-2] == pytest.approx(3 / 7)
        assert obs_3_5[-1] == pytest.approx(5 / 7)

    def test_puck_direction_flag(self, config, physics):
        ai, puck, human = _make_entity_trio(config, physics)
        puck.velocity = [-5.0, 0.0]  # moving left (toward player)
        obs = create_observation(ai, puck, human, 0, 0, model_type="v2")
        assert obs[10] == 1.0  # puck moving to player flag

        puck.velocity = [5.0, 0.0]  # moving right (toward AI)
        obs = create_observation(ai, puck, human, 0, 0, model_type="v2")
        assert obs[10] == 0.0


# ===================================================================
# ModelLoader utilities (no model file needed)
# ===================================================================

from game.ai.model_loader import find_best_model, _load_metadata, _iter_run_dirs


class TestModelLoader:
    def test_no_models_dir(self, tmp_path):
        path, mtype = find_best_model(str(tmp_path))
        assert path is None
        assert mtype is None

    def test_empty_models_dir(self, tmp_path):
        (tmp_path / "models").mkdir()
        path, mtype = find_best_model(str(tmp_path))
        assert path is None

    def test_finds_best_model_by_metadata(self, tmp_path):
        models = tmp_path / "models"
        run_a = models / "run_a" / "best_model"
        run_b = models / "run_b" / "best_model"
        run_a.mkdir(parents=True)
        run_b.mkdir(parents=True)

        # Create dummy model files
        (run_a / "best_model.zip").write_text("dummy")
        (run_b / "best_model.zip").write_text("dummy")

        # Metadata: run_b has better score
        with open(models / "run_a" / "metadata.json", "w") as f:
            json.dump({"best_mean_reward": 5.0}, f)
        with open(models / "run_b" / "metadata.json", "w") as f:
            json.dump({"best_mean_reward": 10.0}, f)

        path, _ = find_best_model(str(tmp_path))
        assert path is not None
        assert "run_b" in path

    def test_load_metadata_json(self, tmp_path):
        run = tmp_path / "run_test"
        run.mkdir()
        with open(run / "metadata.json", "w") as f:
            json.dump({"best_mean_reward": 42.0, "preset": "v2_standard"}, f)

        meta = _load_metadata(str(run))
        assert meta is not None
        assert meta["best_mean_reward"] == 42.0

    def test_load_metadata_missing(self, tmp_path):
        meta = _load_metadata(str(tmp_path))
        assert meta is None

    def test_iter_run_dirs(self, tmp_path):
        models = tmp_path / "models"
        (models / "run_a").mkdir(parents=True)
        (models / "run_b").mkdir(parents=True)
        (models / "file.zip").write_text("not a dir")

        dirs = list(_iter_run_dirs(str(models)))
        assert len(dirs) == 2


# ===================================================================
# Level Mechanics
# ===================================================================

from game.core.mechanics import (
    create_mechanic, NoneMechanic, UVZonesMechanic,
    FogOfWarMechanic, ShrinkingFieldMechanic, HeatWavesMechanic,
)


class TestMechanics:
    def test_factory_none(self, config):
        m = create_mechanic(config, {"mechanics": {"type": "none"}})
        assert isinstance(m, NoneMechanic)

    def test_factory_uv(self, config):
        m = create_mechanic(config, {"mechanics": {"type": "uv_zones", "zone_count": 2}})
        assert isinstance(m, UVZonesMechanic)

    def test_factory_fog(self, config):
        m = create_mechanic(config, {"mechanics": {"type": "fog_of_war"}})
        assert isinstance(m, FogOfWarMechanic)

    def test_factory_shrink(self, config):
        m = create_mechanic(config, {"mechanics": {"type": "shrinking_field"}})
        assert isinstance(m, ShrinkingFieldMechanic)

    def test_factory_heat(self, config):
        m = create_mechanic(config, {"mechanics": {"type": "heat_waves"}})
        assert isinstance(m, HeatWavesMechanic)

    def test_factory_unknown_defaults_none(self, config):
        m = create_mechanic(config, {"mechanics": {"type": "unknown_xyz"}})
        assert isinstance(m, NoneMechanic)

    def test_factory_no_mechanics_key(self, config):
        m = create_mechanic(config, {})
        assert isinstance(m, NoneMechanic)

    def test_none_mechanic_modifiers(self, config):
        m = NoneMechanic(config, {})
        assert m.get_puck_speed_modifier() == 1.0
        assert m.get_friction_modifier() == 1.0
        assert m.get_field_bounds() is None

    def test_uv_zones_speed_boost(self, config, physics):
        m = UVZonesMechanic(config, {"zone_count": 1, "zone_radius": 600,
                                      "speed_boost": 1.5, "zone_move_speed": 0, "zone_pulse_rate": 2})
        puck = Puck(config, physics)
        # Force zone to cover entire field
        m.zones[0] = {"x": config.width / 2, "y": config.height / 2, "dx": 0, "dy": 0}
        m.zone_radius = 600

        puck.velocity = [5.0, 0.0]
        m.update(0.016, puck, [], None)
        assert m.get_puck_speed_modifier() == 1.5

    def test_shrinking_field_bounds(self, config):
        m = ShrinkingFieldMechanic(config, {"shrink_rate": 100, "player_goal_expand": 15,
                                            "ai_goal_shrink": 10, "min_field_ratio": 0.6})
        puck_mock = type("P", (), {"position": [400.0, 250.0], "velocity": [0.0, 0.0],
                                   "radius": 15, "rect": pygame.Rect(385, 235, 30, 30)})()
        m.update(1.0, puck_mock, [], None)
        bounds = m.get_field_bounds()
        assert bounds is not None
        left, top, right, bottom = bounds
        assert left > 0
        assert top > 0

    def test_shrinking_field_on_goal_player(self, config):
        m = ShrinkingFieldMechanic(config, {"shrink_rate": 10, "player_goal_expand": 50,
                                            "ai_goal_shrink": 10, "min_field_ratio": 0.6})
        m.shrink_offset_x = 30.0
        m.on_goal("player")
        assert m.shrink_offset_x < 30.0  # Walls pushed back

    def test_shrinking_field_on_goal_ai(self, config):
        m = ShrinkingFieldMechanic(config, {"shrink_rate": 10, "player_goal_expand": 15,
                                            "ai_goal_shrink": 10, "min_field_ratio": 0.6})
        m.shrink_offset_x = 10.0
        m.on_goal("ai")
        assert m.shrink_offset_x > 10.0  # Walls shrink more

    def test_heat_waves_cycle(self, config):
        m = HeatWavesMechanic(config, {"wave_interval": 1.0, "wave_duration": 0.5,
                                        "friction_reduction": 0.5, "visual_distortion": False,
                                        "trail_enabled": False})
        puck_mock = type("P", (), {"position": [400.0, 250.0], "velocity": [5.0, 0.0],
                                   "radius": 15})()
        # Should not be active initially
        assert not m.wave_active
        assert m.get_friction_modifier() == 1.0

        # Advance past interval
        m.update(1.1, puck_mock, [], None)
        assert m.wave_active
        assert m.get_friction_modifier() == 0.5

        # Advance past duration
        m.update(0.6, puck_mock, [], None)
        assert not m.wave_active
        assert m.get_friction_modifier() == 1.0
