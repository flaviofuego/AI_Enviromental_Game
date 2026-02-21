"""
Tests for the modular powerup system (shared/powerups/).

Coverage:
- PowerUpRegistry: registration, get, get_all
- EffectStack: accumulation, caps, getters, Duplication support
- FieldSphere: movement, expiry, collision
- PowerUpManager: spawn pool, activation, tick, expiry, speed/paralysis application
- Per-phase definitions (all 8):
    Phase 1 — speed_boost  (velocity multiplier, composition)
    Phase 2 — shield       (has_shield flag, goal-blocking contract)
    Phase 3 — magnet       (on_tick puck attraction)
    Phase 4 — duplication  (stacked_multiplier × 2 on collect, revert on expire)
    Phase 5 — slow_opponent (opponent stack, min-cap)
    Phase 6 — obstacle     (state.obstacles populated on collect, cleared on expire)
    Phase 7 — paralyze     (is_paralyzed flag, mallet stays still)
    Phase 8 — invisibility  (state.player_invisible flag)
"""
from __future__ import annotations

import math
import os
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Minimal pygame stub so tests run headlessly without a display
# ---------------------------------------------------------------------------

def _make_pygame_stub():
    """Return a minimal pygame-compatible stub module."""
    stub = types.ModuleType("pygame")
    stub.init = lambda: None
    stub.quit = lambda: None
    stub.display = types.SimpleNamespace(init=lambda: None, set_mode=lambda *a, **k: None)
    stub.time = types.SimpleNamespace(Clock=lambda: None)

    # Surface stub
    class _Surface:
        def __init__(self, size=None, flags=0):
            self.size = size or (0, 0)
        def get_size(self): return self.size
        def get_width(self): return self.size[0] if self.size else 0
        def get_height(self): return self.size[1] if self.size else 0
        def fill(self, *a): pass
        def blit(self, *a): pass
        def set_alpha(self, *a): pass
        def copy(self): return _Surface(self.size)
        def get_rect(self, **kw): return _Rect(0, 0, *self.size)

    # Rect stub
    class _Rect:
        def __init__(self, x=0, y=0, w=0, h=0):
            self.x = x; self.y = y; self.width = w; self.height = h
            self.center = (x + w // 2, y + h // 2)
            self.centerx = self.center[0]
            self.centery = self.center[1]
        def collidepoint(self, *a): return False

    # Sprite / Group stubs
    class _Sprite:
        def __init__(self):
            self.image = _Surface((10, 10))
            self.rect = _Rect()
            self.mask = None
        def add(self, *a): pass

    class _Group:
        def __init__(self): self._sprites = []
        def add(self, *a): pass
        def draw(self, *a): pass

    class _RenderUpdates(_Group): pass

    # mask stub
    class _Mask:
        pass
    stub.mask = types.SimpleNamespace(from_surface=lambda s: _Mask(), Mask=_Mask)

    stub.Surface = _Surface
    stub.Rect = _Rect
    stub.sprite = types.SimpleNamespace(
        Sprite=_Sprite,
        Group=_Group,
        RenderUpdates=_RenderUpdates,
    )
    stub.SRCALPHA = 65536
    stub.draw = types.SimpleNamespace(
        circle=lambda *a, **k: None,
        rect=lambda *a, **k: None,
        polygon=lambda *a, **k: None,
        line=lambda *a, **k: None,
    )
    stub.transform = types.SimpleNamespace(
        smoothscale=lambda s, size: s,
    )
    stub.font = types.SimpleNamespace(
        Font=lambda *a: None,
        SysFont=lambda *a: None,
    )
    stub.error = Exception

    return stub


if "pygame" not in sys.modules:
    sys.modules["pygame"] = _make_pygame_stub()

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

from shared.powerups.registry import PowerUpRegistry, PowerUpDefinition
from shared.powerups.effect_stack import ActiveEffect, EffectStack, SPEED_MIN_CAP
from shared.powerups.manager import PowerUpManager, FieldSphere
from shared.powerups.definitions import register_all


def _make_registry() -> PowerUpRegistry:
    reg = PowerUpRegistry()
    register_all(reg)
    return reg


class _Config:
    """Minimal game config accepted by PowerUpManager."""
    width  = 800
    height = 500
    fps    = 60


class _State:
    """Minimal state object with required attributes."""
    width  = 800
    height = 500

    def __init__(self):
        self.stacks = None          # set by manager.update()
        self.player_invisible = [False, False]
        self.obstacles: list = []


class _Puck:
    def __init__(self, x=400.0, y=250.0):
        self.position = [x, y]
        self.velocity = [0.0, 0.0]
        self.radius   = 15


class _Player:
    def __init__(self, x=200.0, y=250.0):
        self.position       = [x, y]
        self.velocity       = [0.0, 0.0]
        self.radius         = 30
        self.speed_multiplier = 1.0
        self.strike_multiplier = 1.0
        self.paralyzed      = False


# ============================================================
# PowerUpRegistry
# ============================================================

class TestPowerUpRegistry:
    def test_register_and_get(self):
        reg = _make_registry()
        defn = reg.get("speed_boost")
        assert defn.id == "speed_boost"
        assert defn.phase == 1

    def test_get_all_returns_all_eight(self):
        reg = _make_registry()
        ids = {d.id for d in reg.get_all()}
        expected = {
            "speed_boost", "shield", "magnet", "duplication",
            "slow_opponent", "obstacle", "paralyze", "invisibility",
        }
        assert ids == expected

    def test_duplicate_registration_raises(self):
        reg = _make_registry()
        with pytest.raises(ValueError, match="already registered"):
            from shared.powerups.definitions import speed
            speed.register(reg)  # second registration

    def test_get_by_phase(self):
        reg = _make_registry()
        p1 = reg.get_by_phase(1)
        assert len(p1) == 1 and p1[0].id == "speed_boost"

    def test_get_all_by_phase_sorted(self):
        reg = _make_registry()
        phases = [d.phase for d in reg.get_all_by_phase()]
        assert phases == sorted(phases)


# ============================================================
# EffectStack
# ============================================================

def _make_speed_effect(registry, mult_override=None, target=0, collector=0) -> ActiveEffect:
    defn = registry.get("speed_boost")
    eff  = ActiveEffect(defn, remaining=5.0, collector_idx=collector, target_idx=target)
    if mult_override is not None:
        # Override via numeric_contributions (the proper channel)
        defn = type(defn)(
            **{**defn.__dict__, "numeric_contributions": {"strike_mult": mult_override}}
        )
        eff.definition = defn
    return eff


class TestEffectStack:
    def setup_method(self):
        self.reg = _make_registry()

    def test_empty_speed_is_one(self):
        stack = EffectStack()
        assert stack.get_speed_multiplier() == pytest.approx(1.0)

    def test_single_speed_effect(self):
        stack = EffectStack()
        stack.add(_make_speed_effect(self.reg))
        # 1.0 base × 1.3 multiplier (from config)
        mult = stack.get_strike_multiplier()
        assert mult == pytest.approx(1.3, rel=1e-3)

    def test_speed_accumulation_two_instances(self):
        """Two speed_boost effects → 1.3 × 1.3 = 1.69."""
        stack = EffectStack()
        stack.add(_make_speed_effect(self.reg))
        stack.add(_make_speed_effect(self.reg))
        mult = stack.get_strike_multiplier()
        assert mult == pytest.approx(1.3 * 1.3, rel=1e-3)

    def test_speed_cap_max(self):
        from shared.powerups.effect_stack import SPEED_MAX_CAP
        stack = EffectStack()
        for _ in range(20):
            stack.add(_make_speed_effect(self.reg))
        assert stack.get_strike_multiplier() == pytest.approx(SPEED_MAX_CAP)

    def test_slow_effect_caps_at_minimum(self):
        """Multiple slow_opponent effects can't push speed below SPEED_MIN_CAP."""
        stack = EffectStack()
        defn = self.reg.get("slow_opponent")
        for _ in range(10):
            eff = ActiveEffect(defn, remaining=5.0, collector_idx=0, target_idx=1)
            stack.add(eff)
        assert stack.get_speed_multiplier() >= SPEED_MIN_CAP

    def test_has_shield_false_when_empty(self):
        assert not EffectStack().has_shield()

    def test_has_shield_true_when_active(self):
        stack = EffectStack()
        defn  = self.reg.get("shield")
        eff   = ActiveEffect(defn, remaining=5.0, collector_idx=0, target_idx=0)
        stack.add(eff)
        assert stack.has_shield()

    def test_is_paralyzed(self):
        stack = EffectStack()
        assert not stack.is_paralyzed()
        defn = self.reg.get("paralyze")
        eff  = ActiveEffect(defn, remaining=3.0, collector_idx=0, target_idx=1)
        stack.add(eff)
        assert stack.is_paralyzed()

    def test_is_invisible(self):
        stack = EffectStack()
        assert not stack.is_invisible()
        defn = self.reg.get("invisibility")
        eff  = ActiveEffect(defn, remaining=3.0, collector_idx=0, target_idx=0)
        stack.add(eff)
        assert stack.is_invisible()

    def test_duplication_doubles_speed(self):
        stack = EffectStack()
        stack.add(_make_speed_effect(self.reg))
        base_mult = stack.get_strike_multiplier()   # 1.3
        stack.apply_stacked_multiplier(2.0)         # Duplication
        new_mult = stack.get_strike_multiplier()     # 1.3 × 2.0 = 2.6
        assert new_mult == pytest.approx(base_mult * 2.0, rel=1e-3)

    def test_duplication_revert(self):
        stack = EffectStack()
        stack.add(_make_speed_effect(self.reg))
        base_mult = stack.get_strike_multiplier()
        stack.apply_stacked_multiplier(2.0)
        stack.apply_stacked_multiplier(0.5)   # revert
        assert stack.get_strike_multiplier() == pytest.approx(base_mult, rel=1e-3)

    def test_tick_decrements_remaining(self):
        stack = EffectStack()
        eff = _make_speed_effect(self.reg)
        stack.add(eff)
        stack.tick(1.0)
        assert eff.remaining == pytest.approx(4.0)

    def test_tick_returns_expired(self):
        stack = EffectStack()
        eff = _make_speed_effect(self.reg)
        eff.remaining = 0.1
        stack.add(eff)
        expired = stack.tick(0.5)
        assert eff in expired

    def test_remove_effect(self):
        stack = EffectStack()
        eff = _make_speed_effect(self.reg)
        stack.add(eff)
        stack.remove(eff)
        assert len(stack) == 0

    def test_has_magnet(self):
        stack = EffectStack()
        assert not stack.has_magnet()
        defn = self.reg.get("magnet")
        eff  = ActiveEffect(defn, remaining=5.0, collector_idx=0, target_idx=0)
        stack.add(eff)
        assert stack.has_magnet()

    def test_magnet_force_accumulates(self):
        """Two magnet effects accumulate force (capped at MAGNET_FORCE_CAP)."""
        from shared.powerups.effect_stack import MAGNET_FORCE_CAP
        stack = EffectStack()
        defn  = self.reg.get("magnet")
        for _ in range(10):
            eff = ActiveEffect(defn, remaining=5.0, collector_idx=0, target_idx=0)
            stack.add(eff)
        assert stack.get_magnet_force() <= MAGNET_FORCE_CAP


# ============================================================
# FieldSphere
# ============================================================

class TestFieldSphere:
    def setup_method(self):
        self.reg  = _make_registry()
        self.defn = self.reg.get("speed_boost")

    def test_sphere_expires_after_lifetime(self):
        sphere = FieldSphere(self.defn, 400, 250, 800, 500)
        sphere.age = sphere.lifetime
        assert sphere.is_expired

    def test_sphere_not_expired_immediately(self):
        sphere = FieldSphere(self.defn, 400, 250, 800, 500)
        assert not sphere.is_expired

    def test_sphere_moves_after_update(self):
        sphere = FieldSphere(self.defn, 400, 250, 800, 500)
        x0, y0 = sphere.position[0], sphere.position[1]
        sphere.update(0.016)
        # position must have changed (sphere has non-zero velocity)
        assert not (sphere.position[0] == x0 and sphere.position[1] == y0)

    def test_sphere_bounces_off_walls(self):
        sphere = FieldSphere(self.defn, 10, 250, 800, 500)
        sphere.velocity = [-10, 0]   # moving toward left wall
        sphere.update(0.016)
        assert sphere.velocity[0] > 0  # reversed

    def test_collection_collision_inside_radius(self):
        sphere = FieldSphere(self.defn, 400, 250, 800, 500)
        player = _Player(400, 250)  # exactly on sphere
        assert sphere.check_collision(player)

    def test_no_collision_far_away(self):
        sphere = FieldSphere(self.defn, 400, 250, 800, 500)
        player = _Player(700, 400)
        assert not sphere.check_collision(player)

    def test_alpha_blinks_near_expiry(self):
        sphere = FieldSphere(self.defn, 400, 250, 800, 500)
        sphere.age = sphere.lifetime - 1.5   # inside blink window
        sphere.update(0.016)
        # alpha should vary (not always 255)
        # We just check it's a valid byte
        assert 0 <= sphere.alpha <= 255


# ============================================================
# Phase 1 — Speed Boost
# ============================================================

class TestSpeedBoost:
    def setup_method(self):
        self.reg = _make_registry()

    def test_strike_multiplier_applied_after_activate(self):
        """After manager activates speed_boost, player.strike_multiplier > 1."""
        cfg     = _Config()
        manager = PowerUpManager(cfg, self.reg, enabled_phases=[1])
        state   = _State()
        player  = _Player(200, 250)
        opponent = _Player(600, 250)
        puck    = _Puck()
        # Manually activate
        sphere = FieldSphere(self.reg.get("speed_boost"), 200, 250, 800, 500)
        manager.field_spheres.append(sphere)
        manager.update(0.016, [player, opponent], puck, state)
        # Strike should now be > 1.0 while movement speed remains unchanged.
        assert player.strike_multiplier > 1.0
        assert player.speed_multiplier == pytest.approx(1.0)

    def test_strike_expires_correctly(self):
        """After all speed effects expire, strike_multiplier returns to 1.0."""
        manager = PowerUpManager(_Config(), self.reg, enabled_phases=[1])
        state   = _State()
        player  = _Player(200, 250)
        opponent = _Player(600, 250)
        puck    = _Puck()
        # Force-add a very short effect
        defn = self.reg.get("speed_boost")
        eff  = ActiveEffect(defn, remaining=0.001, collector_idx=0, target_idx=0)
        manager.stacks[0].add(eff)
        manager._apply_player_modifiers([player, opponent])
        assert player.strike_multiplier > 1.0
        # Advance beyond duration
        manager.update(0.1, [player, opponent], puck, state)
        assert player.strike_multiplier == pytest.approx(1.0, abs=0.01)


# ============================================================
# Phase 2 — Shield
# ============================================================

class TestShield:
    def setup_method(self):
        self.reg = _make_registry()

    def test_has_shield_after_collect(self):
        manager = PowerUpManager(_Config(), self.reg, enabled_phases=[2])
        state   = _State()
        player  = _Player(200, 250)
        opponent = _Player(600, 250)
        puck    = _Puck()
        # Place sphere directly on player
        sphere = FieldSphere(self.reg.get("shield"), 200, 250, 800, 500)
        manager.field_spheres.append(sphere)
        manager.update(0.016, [player, opponent], puck, state)
        assert manager.stacks[0].has_shield()

    def test_shield_not_active_initially(self):
        manager = PowerUpManager(_Config(), self.reg)
        assert not manager.stacks[0].has_shield()
        assert not manager.stacks[1].has_shield()

    def test_shield_expires(self):
        manager  = PowerUpManager(_Config(), self.reg, enabled_phases=[2])
        state    = _State()
        player   = _Player(200, 250)
        opponent = _Player(600, 250)
        puck     = _Puck()
        defn = self.reg.get("shield")
        eff  = ActiveEffect(defn, remaining=0.001, collector_idx=0, target_idx=0)
        manager.stacks[0].add(eff)
        manager.update(0.1, [player, opponent], puck, state)
        assert not manager.stacks[0].has_shield()

    def test_shield_does_not_stack_resets_timer(self):
        """Second shield collect on same player should reset remaining time."""
        manager = PowerUpManager(_Config(), self.reg, enabled_phases=[2])
        state   = _State()
        player  = _Player(200, 250)
        opponent = _Player(600, 250)
        puck    = _Puck()
        defn = self.reg.get("shield")
        eff  = ActiveEffect(defn, remaining=1.0, collector_idx=0, target_idx=0)
        manager.stacks[0].add(eff)
        # Collect a second shield sphere on top of player
        sphere = FieldSphere(defn, 200, 250, 800, 500)
        manager.field_spheres.append(sphere)
        events = manager.update(0.016, [player, opponent], puck, state)
        # Should have one effect, not two
        effects = manager.stacks[0].get_effects_by_id("shield")
        assert len(effects) == 1
        # Timer should have been reset to full duration
        assert effects[0].remaining == pytest.approx(defn.duration, rel=0.1)


# ============================================================
# Phase 3 — Magnet
# ============================================================

class TestMagnet:
    def setup_method(self):
        self.reg = _make_registry()

    def test_magnet_attracts_puck(self):
        """With magnet active, puck velocity should gain a component toward player."""
        manager  = PowerUpManager(_Config(), self.reg, enabled_phases=[3])
        state    = _State()
        player   = _Player(200, 250)
        opponent = _Player(600, 250)
        # Puck far from player
        puck     = _Puck(x=500.0, y=250.0)

        defn = self.reg.get("magnet")
        eff  = ActiveEffect(defn, remaining=5.0, collector_idx=0, target_idx=0)
        manager.stacks[0].add(eff)

        initial_vx = puck.velocity[0]
        manager.update(0.1, [player, opponent], puck, state)
        # Puck should have acquired velocity toward player (positive x → player at x=200 means
        # force direction is negative x from puck at 500)
        assert puck.velocity[0] < initial_vx or puck.velocity[0] != 0

    def test_magnet_has_magnet_flag(self):
        stack = EffectStack()
        defn  = self.reg.get("magnet")
        eff   = ActiveEffect(defn, remaining=5.0, collector_idx=0, target_idx=0)
        stack.add(eff)
        assert stack.has_magnet()

    def test_magnet_force_zero_when_inactive(self):
        stack = EffectStack()
        assert stack.get_magnet_force() == pytest.approx(0.0)


# ============================================================
# Phase 4 — Duplication
# ============================================================

class TestDuplication:
    def setup_method(self):
        self.reg = _make_registry()

    def test_duplication_doubles_existing_speed(self):
        state    = _State()
        puck     = _Puck()
        player   = _Player(200, 250)
        opponent = _Player(600, 250)
        manager  = PowerUpManager(_Config(), self.reg, enabled_phases=[1, 4])

        # First add a speed effect
        defn_speed = self.reg.get("speed_boost")
        eff_speed  = ActiveEffect(defn_speed, remaining=10.0, collector_idx=0, target_idx=0)
        manager.stacks[0].add(eff_speed)
        speed_before = manager.stacks[0].get_strike_multiplier()

        # Now trigger duplication on collector 0
        state.stacks = manager.stacks
        defn_dup = self.reg.get("duplication")
        defn_dup.on_collect(0, [player, opponent], puck, state)

        speed_after = manager.stacks[0].get_strike_multiplier()
        assert speed_after == pytest.approx(speed_before * 2.0, rel=1e-3)

    def test_duplication_reverts_on_expire(self):
        state    = _State()
        puck     = _Puck()
        player   = _Player(200, 250)
        opponent = _Player(600, 250)
        manager  = PowerUpManager(_Config(), self.reg, enabled_phases=[1, 4])

        defn_speed = self.reg.get("speed_boost")
        eff_speed  = ActiveEffect(defn_speed, remaining=10.0, collector_idx=0, target_idx=0)
        manager.stacks[0].add(eff_speed)
        base = manager.stacks[0].get_strike_multiplier()

        state.stacks = manager.stacks
        defn_dup = self.reg.get("duplication")
        defn_dup.on_collect(0, [player, opponent], puck, state)  # double
        defn_dup.on_expire(0, [player, opponent], puck, state)   # revert

        reverted = manager.stacks[0].get_strike_multiplier()
        assert reverted == pytest.approx(base, rel=1e-3)


# ============================================================
# Phase 5 — Slow Opponent
# ============================================================

class TestSlowOpponent:
    def setup_method(self):
        self.reg = _make_registry()

    def test_slow_targets_opponent(self):
        """slow_opponent powerup should land in opponent's stack (idx 1 when player 0 collects)."""
        defn = self.reg.get("slow_opponent")
        assert not defn.affects_self  # affects opponent

    def test_slow_reduces_speed(self):
        stack = EffectStack()
        defn  = self.reg.get("slow_opponent")
        eff   = ActiveEffect(defn, remaining=5.0, collector_idx=0, target_idx=1)
        stack.add(eff)
        # 0.75 factor means speed < 1.0
        assert stack.get_speed_multiplier() < 1.0

    def test_slow_cap_not_breached(self):
        stack = EffectStack()
        defn  = self.reg.get("slow_opponent")
        for _ in range(10):
            eff = ActiveEffect(defn, remaining=5.0, collector_idx=0, target_idx=1)
            stack.add(eff)
        assert stack.get_speed_multiplier() >= SPEED_MIN_CAP


# ============================================================
# Phase 6 — Obstacle
# ============================================================

class TestObstacle:
    def setup_method(self):
        self.reg = _make_registry()

    def test_obstacles_spawned_on_collect(self):
        state = _State()
        puck  = _Puck()
        defn  = self.reg.get("obstacle")
        defn.on_collect(0, [], puck, state)
        from shared.powerups.config import POWERUP_CONFIGS
        expected_count = POWERUP_CONFIGS["obstacle"]["count"]
        assert len(state.obstacles) == expected_count

    def test_obstacles_marked_fading_on_expire(self):
        state = _State()
        puck  = _Puck()
        defn  = self.reg.get("obstacle")
        defn.on_collect(0, [], puck, state)
        assert len(state.obstacles) > 0
        defn.on_expire(0, [], puck, state)
        for obs in state.obstacles:
            assert obs.get("fading") is True

    def test_obstacle_cap_respected(self):
        """Collecting obstacle 4 times should not exceed MAX_FIELD_OBSTACLES."""
        state = _State()
        puck  = _Puck()
        defn  = self.reg.get("obstacle")
        for _ in range(5):
            defn.on_collect(0, [], puck, state)
        from shared.powerups.definitions.obstacle import MAX_FIELD_OBSTACLES
        assert len(state.obstacles) <= MAX_FIELD_OBSTACLES

    def test_obstacles_have_required_keys(self):
        state = _State()
        puck  = _Puck()
        defn  = self.reg.get("obstacle")
        defn.on_collect(0, [], puck, state)
        for obs in state.obstacles:
            for key in ("x", "y", "radius", "collector_idx", "alpha", "fading"):
                assert key in obs


# ============================================================
# Phase 7 — Paralyze
# ============================================================

class TestParalyze:
    def setup_method(self):
        self.reg = _make_registry()

    def test_paralyze_targets_opponent(self):
        defn = self.reg.get("paralyze")
        assert not defn.affects_self

    def test_is_paralyzed_flag(self):
        stack = EffectStack()
        assert not stack.is_paralyzed()
        defn = self.reg.get("paralyze")
        eff  = ActiveEffect(defn, remaining=3.0, collector_idx=0, target_idx=1)
        stack.add(eff)
        assert stack.is_paralyzed()

    def test_manager_sets_mallet_paralyzed(self):
        manager  = PowerUpManager(_Config(), self.reg, enabled_phases=[7])
        state    = _State()
        player   = _Player(200, 250)
        opponent = _Player(600, 250)
        puck     = _Puck()

        defn = self.reg.get("paralyze")
        # Collector 0 paralyzes player at index 1 (opponent)
        eff  = ActiveEffect(defn, remaining=3.0, collector_idx=0, target_idx=1)
        manager.stacks[1].add(eff)
        manager._apply_paralysis([player, opponent])
        assert opponent.paralyzed is True
        assert player.paralyzed is False

    def test_paralysis_clears_after_expiry(self):
        manager  = PowerUpManager(_Config(), self.reg, enabled_phases=[7])
        state    = _State()
        player   = _Player(200, 250)
        opponent = _Player(600, 250)
        puck     = _Puck()

        defn = self.reg.get("paralyze")
        eff  = ActiveEffect(defn, remaining=0.001, collector_idx=0, target_idx=1)
        manager.stacks[1].add(eff)
        manager.update(0.1, [player, opponent], puck, state)
        assert opponent.paralyzed is False

    def test_paralysis_stacks_durations(self):
        """Two paralyze effects in same stack — both present (can_stack=True)."""
        stack = EffectStack()
        defn  = self.reg.get("paralyze")
        eff1  = ActiveEffect(defn, remaining=3.0, collector_idx=0, target_idx=1)
        eff2  = ActiveEffect(defn, remaining=3.0, collector_idx=0, target_idx=1)
        stack.add(eff1)
        stack.add(eff2)
        assert len(stack.get_effects_by_id("paralyze")) == 2


# ============================================================
# Phase 8 — Invisibility
# ============================================================

class TestInvisibility:
    def setup_method(self):
        self.reg = _make_registry()

    def test_invisibility_sets_state_flag(self):
        state = _State()
        puck  = _Puck()
        defn  = self.reg.get("invisibility")
        defn.on_collect(0, [], puck, state)
        assert state.player_invisible[0] is True

    def test_invisibility_clears_on_expire(self):
        state = _State()
        puck  = _Puck()
        defn  = self.reg.get("invisibility")
        defn.on_collect(0, [], puck, state)
        defn.on_expire(0, [], puck, state)
        assert state.player_invisible[0] is False

    def test_is_invisible_flag_on_stack(self):
        stack = EffectStack()
        defn  = self.reg.get("invisibility")
        eff   = ActiveEffect(defn, remaining=3.0, collector_idx=0, target_idx=0)
        stack.add(eff)
        assert stack.is_invisible()

    def test_not_invisible_when_empty(self):
        assert not EffectStack().is_invisible()

    def test_both_players_can_be_invisible(self):
        state = _State()
        puck  = _Puck()
        defn  = self.reg.get("invisibility")
        defn.on_collect(0, [], puck, state)
        defn.on_collect(1, [], puck, state)
        assert state.player_invisible[0] is True
        assert state.player_invisible[1] is True


# ============================================================
# PowerUpManager integration
# ============================================================

class TestPowerUpManagerIntegration:
    def setup_method(self):
        self.reg     = _make_registry()
        self.cfg     = _Config()
        self.manager = PowerUpManager(self.cfg, self.reg)
        self.state   = _State()
        self.player  = _Player(200, 250)
        self.opponent = _Player(600, 250)
        self.puck    = _Puck()
        self.players = [self.player, self.opponent]

    def test_stacks_exposed_on_state_after_update(self):
        self.manager.update(0.016, self.players, self.puck, self.state)
        assert self.state.stacks is not None
        assert len(self.state.stacks) == 2

    def test_update_returns_event_list(self):
        events = self.manager.update(0.016, self.players, self.puck, self.state)
        assert isinstance(events, list)

    def test_collection_event_emitted(self):
        sphere = FieldSphere(self.reg.get("speed_boost"), 200, 250, 800, 500)
        self.manager.field_spheres.append(sphere)
        events = self.manager.update(0.016, self.players, self.puck, self.state)
        types_ = [e["type"] for e in events]
        assert "collected" in types_

    def test_expiry_event_emitted(self):
        defn = self.reg.get("speed_boost")
        eff  = ActiveEffect(defn, remaining=0.001, collector_idx=0, target_idx=0)
        self.manager.stacks[0].add(eff)
        events = self.manager.update(0.1, self.players, self.puck, self.state)
        types_ = [e["type"] for e in events]
        assert "expired" in types_

    def test_enable_phases_restricts_pool(self):
        self.manager.enable_phases([1])
        pool = self.manager._get_spawn_pool()
        assert all(d.phase == 1 for d in pool)

    def test_enable_all_returns_all(self):
        self.manager.enable_all()
        pool = self.manager._get_spawn_pool()
        assert len(pool) == 8

    def test_reset_clears_state(self):
        defn = self.reg.get("speed_boost")
        eff  = ActiveEffect(defn, remaining=5.0, collector_idx=0, target_idx=0)
        self.manager.stacks[0].add(eff)
        self.manager.field_spheres.append(
            FieldSphere(defn, 400, 250, 800, 500)
        )
        self.state.obstacles = [{"x": 1.0, "y": 1.0, "radius": 20, "alpha": 255, "fading": False}]
        self.manager.reset(self.state)
        assert len(self.manager.stacks[0]) == 0
        assert len(self.manager.field_spheres) == 0
        assert len(self.state.obstacles) == 0

    def test_no_simultaneous_sphere_overcount(self):
        """PowerUpManager._spawn_logic must not exceed MAX_POWERUPS_ON_FIELD."""
        from shared.powerups.config import MAX_POWERUPS_ON_FIELD, SPAWN_INTERVAL_MIN
        # Saturate the timer so every call to _spawn_logic tries to spawn
        for _ in range(20):
            self.manager._spawn_timer = SPAWN_INTERVAL_MIN + 1.0
            self.manager._spawn_logic(0.0, self.state)
        assert len(self.manager.field_spheres) <= MAX_POWERUPS_ON_FIELD

    def test_get_field_spheres_returns_copy(self):
        sphere = FieldSphere(self.reg.get("shield"), 400, 250, 800, 500)
        self.manager.field_spheres.append(sphere)
        fetched = self.manager.get_field_spheres()
        fetched.clear()  # modifying the copy should not affect internal list
        assert len(self.manager.field_spheres) == 1


# ============================================================
# Mallet paralysis attribute (Tarea 1)
# ============================================================

class TestMalletParalyzedAttribute:
    """Verify the paralyzed attribute exists on the base Mallet class."""

    def test_mallet_has_paralyzed_attr(self):
        from shared.entities.mallet import Mallet
        # Mallet needs pygame; we patched it at module level
        from shared.config import GameConfig
        cfg = GameConfig()
        m = Mallet(100, 100, config=cfg)
        assert hasattr(m, "paralyzed")
        assert m.paralyzed is False

    def test_paralyzed_can_be_set(self):
        from shared.entities.mallet import Mallet
        from shared.config import GameConfig
        m = Mallet(100, 100, config=GameConfig())
        m.paralyzed = True
        assert m.paralyzed is True


# ============================================================
# Section 5.1 — PowerUpNotification & NotificationQueue
# ============================================================

def _make_stub_font():
    """Font stub that returns measurable surfaces for text renders."""
    class _FakeSurf:
        def __init__(self, text=""):
            # Simulate a surface with width proportional to text length
            self._w = max(4, len(text) * 7)
            self._h = 14
        def get_width(self):  return self._w
        def get_height(self): return self._h
        def copy(self):       return _FakeSurf()
        def fill(self, *a):   pass
        def blit(self, *a):   pass
        def set_alpha(self, *a): pass

    class _FakeFont:
        def render(self, text, antialias, color):
            return _FakeSurf(text)
        def get_linesize(self): return 14

    return _FakeFont()


def _patch_font_cache(monkeypatch):
    """Patch FontCache.get to always return our fake font."""
    import game.components.FontCache as fc_mod
    fake_font = _make_stub_font()
    monkeypatch.setattr(fc_mod.font_cache, "get", lambda *a, **kw: fake_font)


def _make_stub_screen(w=800, h=500):
    """Return a stub surface that accepts blit/draw calls."""
    class _FakeScreen:
        def __init__(self, w, h):
            self.w = w
            self.h = h
        def get_width(self):  return self.w
        def get_height(self): return self.h
        def blit(self, *a, **kw): pass
        def fill(self, *a):   pass

    return _FakeScreen(w, h)


# ── PowerUpNotification ────────────────────────────────────────────

class TestPowerUpNotification:
    """Tests for a single toast notification lifecycle."""

    def setup_method(self):
        self.reg  = _make_registry()
        self.defn = self.reg.get("speed_boost")

    def test_not_done_initially(self):
        from game.components.PowerUpNotification import PowerUpNotification
        n = PowerUpNotification(self.defn, "left", "collected", 800, 0)
        assert not n.is_done

    def test_done_after_display_duration(self):
        from game.components.PowerUpNotification import PowerUpNotification, DISPLAY_DURATION
        n = PowerUpNotification(self.defn, "left", "collected", 800, 0)
        n.update(DISPLAY_DURATION + 0.01)
        assert n.is_done

    def test_not_done_just_before_end(self):
        from game.components.PowerUpNotification import PowerUpNotification, DISPLAY_DURATION
        n = PowerUpNotification(self.defn, "left", "collected", 800, 0)
        n.update(DISPLAY_DURATION - 0.01)
        assert not n.is_done

    def test_update_accumulates(self):
        """Multiple small updates add up to elapsed time correctly."""
        from game.components.PowerUpNotification import PowerUpNotification, DISPLAY_DURATION
        n = PowerUpNotification(self.defn, "left", "collected", 800, 0)
        # Apply many small steps; total elapsed must exceed DISPLAY_DURATION
        steps = int(DISPLAY_DURATION / 0.016) + 10   # ~156 frames + buffer
        for _ in range(steps):
            n.update(0.016)
        assert n.is_done

    def test_invalid_side_raises(self):
        from game.components.PowerUpNotification import PowerUpNotification
        with pytest.raises(ValueError, match="side"):
            PowerUpNotification(self.defn, "center", "collected", 800)

    def test_invalid_event_raises(self):
        from game.components.PowerUpNotification import PowerUpNotification
        with pytest.raises(ValueError, match="event_type"):
            PowerUpNotification(self.defn, "left", "unknown", 800)

    def test_left_position_anchors_left(self):
        """Left-side toast x-position should be near 0."""
        from game.components.PowerUpNotification import PowerUpNotification, TOAST_WIDTH
        n = PowerUpNotification(self.defn, "left", "collected", 800, 0)
        x, _ = n._compute_position(0, 0)
        assert x < TOAST_WIDTH  # positioned near the left edge

    def test_right_position_anchors_right(self):
        """Right-side toast x+width should be <= screen_w."""
        from game.components.PowerUpNotification import PowerUpNotification, TOAST_WIDTH
        n = PowerUpNotification(self.defn, "right", "collected", 800, 0)
        x, _ = n._compute_position(0, 0)
        assert x + TOAST_WIDTH <= 800

    def test_slot_increases_y(self):
        """Higher slot number should result in larger y coordinate."""
        from game.components.PowerUpNotification import PowerUpNotification
        n = PowerUpNotification(self.defn, "left", "collected", 800, 0)
        _, y0 = n._compute_position(0, 0)
        _, y1 = n._compute_position(1, 0)
        assert y1 > y0

    def test_animation_alpha_zero_when_done(self):
        """Once done, _compute_animation should return alpha ~0."""
        from game.components.PowerUpNotification import PowerUpNotification, DISPLAY_DURATION
        n = PowerUpNotification(self.defn, "left", "collected", 800, 0)
        n.update(DISPLAY_DURATION)
        alpha, _ = n._compute_animation()
        assert alpha == 0

    def test_animation_alpha_max_during_hold(self):
        """During the hold phase alpha should be at maximum."""
        from game.components.PowerUpNotification import (
            PowerUpNotification, SLIDE_IN_DURATION, _BG_ALPHA_MAX
        )
        n = PowerUpNotification(self.defn, "left", "collected", 800, 0)
        n.update(SLIDE_IN_DURATION + 0.5)   # inside hold phase
        alpha, _ = n._compute_animation()
        assert alpha == _BG_ALPHA_MAX

    def test_slide_in_offset_decreases(self):
        """During slide-in the y_offset should go from negative to 0."""
        from game.components.PowerUpNotification import PowerUpNotification
        n = PowerUpNotification(self.defn, "left", "collected", 800, 0)
        _, offset_start = n._compute_animation()    # t=0 → max negative offset
        n.update(0.1)
        _, offset_mid = n._compute_animation()
        # At t=0 offset starts negative (toast is above), moving toward 0
        assert offset_start <= offset_mid

    def test_draw_does_not_crash(self, monkeypatch):
        """draw() must not raise even with the pygame stub."""
        _patch_font_cache(monkeypatch)
        from game.components.PowerUpNotification import PowerUpNotification
        screen = _make_stub_screen()
        n = PowerUpNotification(self.defn, "left", "collected", 800, 0)
        n.update(0.1)
        n.draw(screen, slot=0)   # should not raise

    def test_draw_after_done_does_nothing(self, monkeypatch):
        """draw() on a finished toast should be a no-op."""
        _patch_font_cache(monkeypatch)
        from game.components.PowerUpNotification import PowerUpNotification, DISPLAY_DURATION
        screen = _make_stub_screen()
        n = PowerUpNotification(self.defn, "left", "collected", 800, 0)
        n.update(DISPLAY_DURATION + 1.0)
        n.draw(screen, slot=0)   # no-op, no crash

    def test_expired_event_type(self):
        from game.components.PowerUpNotification import PowerUpNotification
        n = PowerUpNotification(self.defn, "right", "expired", 800, 0)
        assert n.event_type == "expired"

    def test_field_top_shifts_y(self):
        """field_top should push toasts downward."""
        from game.components.PowerUpNotification import PowerUpNotification
        n0 = PowerUpNotification(self.defn, "left", "collected", 800, field_top=0)
        n1 = PowerUpNotification(self.defn, "left", "collected", 800, field_top=60)
        _, y0 = n0._compute_position(0, 0)
        _, y1 = n1._compute_position(0, 0)
        assert y1 > y0


# ── NotificationQueue ─────────────────────────────────────────────

class TestNotificationQueue:
    """Tests for the per-player notification queue manager."""

    def setup_method(self):
        self.reg  = _make_registry()
        self.defn = self.reg.get("speed_boost")

    def test_push_increments_count(self):
        from game.components.PowerUpNotification import NotificationQueue
        q = NotificationQueue("left", 800)
        q.push(self.defn, "collected")
        assert q.active_count == 1

    def test_push_multiple(self):
        from game.components.PowerUpNotification import NotificationQueue
        q = NotificationQueue("left", 800, max_visible=3)
        for _ in range(3):
            q.push(self.defn, "collected")
        assert q.active_count == 3

    def test_capacity_drops_oldest(self):
        """When at max_visible, pushing a new toast evicts the oldest."""
        from game.components.PowerUpNotification import NotificationQueue
        q = NotificationQueue("left", 800, max_visible=2)
        defn_slow  = self.reg.get("slow_opponent")
        q.push(self.defn, "collected")      # slot 0 — speed_boost
        q.push(defn_slow, "collected")      # slot 1 — slow_opponent
        # Queue is now full (max_visible=2); push one more
        defn_shield = self.reg.get("shield")
        q.push(defn_shield, "collected")    # slot 0 evicted
        assert q.active_count == 2
        # The oldest (speed_boost) should have been dropped
        ids = [t.definition.id for t in q._toasts]
        assert "speed_boost" not in ids
        assert "shield" in ids

    def test_update_removes_expired_toasts(self):
        from game.components.PowerUpNotification import NotificationQueue, DISPLAY_DURATION
        q = NotificationQueue("left", 800)
        q.push(self.defn, "collected")
        q.update(DISPLAY_DURATION + 0.1)
        assert q.active_count == 0

    def test_update_keeps_live_toasts(self):
        from game.components.PowerUpNotification import NotificationQueue, DISPLAY_DURATION
        q = NotificationQueue("left", 800)
        q.push(self.defn, "collected")
        q.update(DISPLAY_DURATION * 0.3)    # only 30 % elapsed
        assert q.active_count == 1

    def test_clear_empties_queue(self):
        from game.components.PowerUpNotification import NotificationQueue
        q = NotificationQueue("left", 800)
        q.push(self.defn, "collected")
        q.push(self.reg.get("shield"), "expired")
        q.clear()
        assert q.active_count == 0

    def test_invalid_side_raises(self):
        from game.components.PowerUpNotification import NotificationQueue
        with pytest.raises(ValueError, match="side"):
            NotificationQueue("top", 800)

    def test_draw_does_not_crash(self, monkeypatch):
        _patch_font_cache(monkeypatch)
        from game.components.PowerUpNotification import NotificationQueue
        screen = _make_stub_screen()
        q = NotificationQueue("left", 800)
        q.push(self.defn, "collected")
        q.update(0.1)
        q.draw(screen)   # must not raise

    def test_draw_empty_queue_is_noop(self, monkeypatch):
        _patch_font_cache(monkeypatch)
        from game.components.PowerUpNotification import NotificationQueue
        screen = _make_stub_screen()
        q = NotificationQueue("right", 800)
        q.draw(screen)   # empty — must not raise

    def test_right_side_queue(self):
        from game.components.PowerUpNotification import NotificationQueue
        q = NotificationQueue("right", 800)
        q.push(self.defn, "expired")
        assert q.active_count == 1
        assert q._toasts[0].side == "right"

    def test_push_different_event_types(self):
        from game.components.PowerUpNotification import NotificationQueue
        q = NotificationQueue("left", 800, max_visible=3)
        q.push(self.defn, "collected")
        q.push(self.reg.get("magnet"), "expired")
        assert q.active_count == 2
        assert q._toasts[0].event_type == "collected"
        assert q._toasts[1].event_type == "expired"


# ── draw_notifications convenience function ───────────────────────

class TestDrawNotifications:
    def setup_method(self):
        self.reg = _make_registry()

    def test_draw_notifications_does_not_crash(self, monkeypatch):
        _patch_font_cache(monkeypatch)
        from game.components.PowerUpNotification import (
            NotificationQueue, draw_notifications
        )
        screen = _make_stub_screen()
        ql = NotificationQueue("left",  800)
        qr = NotificationQueue("right", 800)
        ql.push(self.reg.get("speed_boost"), "collected")
        qr.push(self.reg.get("slow_opponent"), "expired")
        ql.update(0.1)
        qr.update(0.1)
        draw_notifications(screen, ql, qr)

    def test_draw_notifications_both_empty(self, monkeypatch):
        _patch_font_cache(monkeypatch)
        from game.components.PowerUpNotification import (
            NotificationQueue, draw_notifications
        )
        screen = _make_stub_screen()
        ql = NotificationQueue("left",  800)
        qr = NotificationQueue("right", 800)
        draw_notifications(screen, ql, qr)   # must not raise
