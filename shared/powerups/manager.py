"""
PowerUpManager — generic, definition-agnostic powerup engine.

Responsibilities (Single Responsibility per method):
    _spawn_logic        — timer-based sphere spawning
    _check_collection   — collision detection between mallets and field spheres
    _activate           — create ActiveEffect + call on_collect + emit events
    _tick_active        — advance timers; call on_tick for each active effect
    _expire_effects     — call on_expire + remove from EffectStack
    _apply_speed        — read EffectStack and push speed_multiplier to players
    _apply_paralysis    — freeze mallet if stack.is_paralyzed()

The manager is framework-agnostic: it works in game and training contexts alike.
Pygame-specific rendering lives in game/entities/powerup_renderer.py (not here).

Usage:
    registry = PowerUpRegistry()
    register_all(registry)
    manager = PowerUpManager(game_config, registry)
    manager.enable_phases([1, 2, 3])

    # Inside game loop:
    events = manager.update(dt, players, puck, state)
    for ev in events:
        if ev["type"] == "collected":
            audio_manager.play(ev["sound_key"])
"""
from __future__ import annotations

import math
import random
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from shared.powerups.registry import PowerUpRegistry, PowerUpDefinition
from shared.powerups.effect_stack import ActiveEffect, EffectStack
from shared.powerups.config import (
    SPHERE_FIELD_LIFETIME,
    SPHERE_SPEED,
    SPHERE_DIRECTION_CHANGE,
    MAX_POWERUPS_ON_FIELD,
    SPAWN_INTERVAL_MIN,
    SPAWN_INTERVAL_MAX,
)

if TYPE_CHECKING:
    from shared.config import GameConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight field-sphere data (no pygame dependency)
# ---------------------------------------------------------------------------

class FieldSphere:
    """
    Data representation of a powerup sphere on the field.

    Visual rendering is delegated to PowerUpSphere in game/entities/powerup_renderer.py.
    This class only holds physics/logic state so it can be used in training
    environments without importing pygame.
    """

    def __init__(
        self,
        definition: PowerUpDefinition,
        x: float,
        y: float,
        field_w: float,
        field_h: float,
    ) -> None:
        self.definition    = definition
        self.position      = [x, y]
        self.radius        = 18  # logical collision radius (px at 800×500)
        self.age           = 0.0
        self.lifetime      = SPHERE_FIELD_LIFETIME
        self.alpha         = 255  # used by renderer
        # Random initial velocity
        angle = random.uniform(0, math.tau)
        speed = SPHERE_SPEED
        self.velocity      = [math.cos(angle) * speed, math.sin(angle) * speed]
        self._dir_timer    = 0.0
        self._field_w      = field_w
        self._field_h      = field_h

    # --- Update -----------------------------------------------------------

    def update(self, dt: float) -> None:
        self.age += dt
        self._dir_timer += dt

        # Random direction change every SPHERE_DIRECTION_CHANGE ± 0.5 s
        if self._dir_timer >= SPHERE_DIRECTION_CHANGE + random.uniform(-0.5, 0.5):
            angle = random.uniform(0, math.tau)
            speed = SPHERE_SPEED
            self.velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            self._dir_timer = 0.0

        # Move
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

        # Bounce off internal field walls (not goal mouths — simplified)
        r = self.radius
        if self.position[0] - r < 0:
            self.position[0] = r
            self.velocity[0] *= -1
        elif self.position[0] + r > self._field_w:
            self.position[0] = self._field_w - r
            self.velocity[0] *= -1
        if self.position[1] - r < 0:
            self.position[1] = r
            self.velocity[1] *= -1
        elif self.position[1] + r > self._field_h:
            self.position[1] = self._field_h - r
            self.velocity[1] *= -1

        # Warning blink (alpha controlled; renderer reads it)
        warn_start = self.lifetime - 2.0
        if self.age > warn_start:
            t = (self.age - warn_start) / 2.0
            self.alpha = int(255 * abs(math.sin(t * math.pi * 8)))
        else:
            self.alpha = 255

    # --- Queries ----------------------------------------------------------

    @property
    def is_expired(self) -> bool:
        return self.age >= self.lifetime

    def check_collision(self, entity) -> bool:
        """Return True if *entity* (a mallet) overlaps this sphere."""
        dx = self.position[0] - entity.position[0]
        dy = self.position[1] - entity.position[1]
        return math.hypot(dx, dy) <= self.radius + entity.radius

    def __repr__(self) -> str:
        return f"FieldSphere(id={self.definition.id!r}, age={self.age:.1f}s)"


# ---------------------------------------------------------------------------
# PowerUpManager
# ---------------------------------------------------------------------------

class PowerUpManager:
    """
    Generic powerup engine — knows nothing about specific powerup logic.

    All powerup behaviour is dispatched through PowerUpDefinition callables
    (on_collect, on_expire, on_tick).

    Returns an event list from ``update()`` so callers (game engine, training
    env) can react to collection/expiry without tight coupling.
    """

    def __init__(
        self,
        config: "GameConfig",
        registry: PowerUpRegistry,
        enabled_phases: Optional[List[int]] = None,
    ) -> None:
        self.config   = config
        self.registry = registry

        # One EffectStack per player slot (always 2 in air hockey)
        self.stacks: List[EffectStack] = [EffectStack(), EffectStack()]

        # Spheres currently on the field
        self.field_spheres: List[FieldSphere] = []

        # Spawn timing
        self._spawn_timer    = 0.0
        self._next_spawn_at  = random.uniform(SPAWN_INTERVAL_MIN, SPAWN_INTERVAL_MAX)

        # Which phases are active (None = all registered)
        self._enabled_ids: Optional[List[str]] = None
        if enabled_phases is not None:
            self.enable_phases(enabled_phases)

    # --- Configuration ----------------------------------------------------

    def enable_phases(self, phases: List[int]) -> None:
        """Restrict spawning to powerups from *phases* only."""
        ids = []
        for defn in self.registry.get_all():
            if defn.phase in phases:
                ids.append(defn.id)
        self._enabled_ids = ids
        logger.debug("PowerUpManager phases enabled: %s", ids)

    def enable_all(self) -> None:
        """Allow all registered powerup types to spawn."""
        self._enabled_ids = None

    # --- Main update (call once per frame) --------------------------------

    def update(
        self,
        dt: float,
        players: list,
        puck: Any,
        state: Any,
    ) -> List[Dict]:
        """
        Advance the powerup simulation by *dt* seconds.

        Parameters
        ----------
        dt      : seconds since last frame
        players : list of mallet entities (index 0 = player 1, 1 = player 2)
        puck    : puck entity
        state   : game state ― must expose .width, .height; may be mutated
                  by on_collect / on_expire callbacks (e.g. state.obstacles)

        Returns
        -------
        List of event dicts with keys: type, powerup_id, player_idx, [sound_key]
        """
        # Expose stacks on state so on_collect/on_expire callbacks can access them
        state.stacks = self.stacks

        events: List[Dict] = []

        self._spawn_logic(dt, state)
        self._update_field_spheres(dt)
        events += self._check_collection(players, puck, state)
        # Advance timers first, then call on_tick, then expire
        self._advance_timers(dt)
        self._tick_active(dt, players, puck, state)
        events += self._do_expire_pass(players, puck, state)
        self._apply_speed(players)
        self._apply_paralysis(players)

        return events

    # --- Spawn logic ------------------------------------------------------

    def _spawn_logic(self, dt: float, state: Any) -> None:
        self._spawn_timer += dt
        if (
            self._spawn_timer >= self._next_spawn_at
            and len(self.field_spheres) < MAX_POWERUPS_ON_FIELD
        ):
            self._spawn_random(state)
            self._spawn_timer = 0.0
            self._next_spawn_at = random.uniform(SPAWN_INTERVAL_MIN, SPAWN_INTERVAL_MAX)

    def _spawn_random(self, state: Any) -> Optional[FieldSphere]:
        """Pick a random eligible definition and place a sphere on the field."""
        pool = self._get_spawn_pool()
        if not pool:
            return None

        defn = random.choice(pool)
        W = getattr(state, "width",  self.config.width)
        H = getattr(state, "height", self.config.height)

        # Spawn in the middle 50 % of the field to avoid goals and edges
        x = random.uniform(W * 0.25, W * 0.75)
        y = random.uniform(H * 0.15, H * 0.85)

        sphere = FieldSphere(defn, x, y, W, H)
        self.field_spheres.append(sphere)
        logger.debug("Spawned %s at (%.0f, %.0f)", defn.id, x, y)
        return sphere

    def _get_spawn_pool(self) -> List[PowerUpDefinition]:
        """Return the list of definitions eligible for spawning."""
        all_defns = self.registry.get_all()
        if self._enabled_ids is None:
            return all_defns
        return [d for d in all_defns if d.id in self._enabled_ids]

    # --- Field sphere lifecycle -------------------------------------------

    def _update_field_spheres(self, dt: float) -> None:
        expired = [s for s in self.field_spheres if s.is_expired]
        for s in expired:
            self.field_spheres.remove(s)
        for s in self.field_spheres:
            s.update(dt)

    # --- Collection detection --------------------------------------------

    def _check_collection(
        self,
        players: list,
        puck: Any,
        state: Any,
    ) -> List[Dict]:
        events: List[Dict] = []
        collected_spheres: List[FieldSphere] = []

        for player_idx, player in enumerate(players):
            for sphere in self.field_spheres:
                if sphere in collected_spheres:
                    continue
                if sphere.check_collision(player):
                    ev = self._activate(sphere, player_idx, players, puck, state)
                    events.append(ev)
                    collected_spheres.append(sphere)

        for sphere in collected_spheres:
            if sphere in self.field_spheres:
                self.field_spheres.remove(sphere)

        return events

    # --- Activation -------------------------------------------------------

    def _activate(
        self,
        sphere: FieldSphere,
        collector_idx: int,
        players: list,
        puck: Any,
        state: Any,
    ) -> Dict:
        defn = sphere.definition
        target_idx = collector_idx if defn.affects_self else (1 - collector_idx)

        # Handle non-stacking: reset timer if already active
        if not defn.can_stack:
            existing = self.stacks[target_idx].get_effects_by_id(defn.id)
            if existing:
                for eff in existing:
                    eff.remaining = defn.duration
                logger.debug("Reset timer for %s on player %d", defn.id, target_idx)
                return {
                    "type":       "reset",
                    "powerup_id": defn.id,
                    "player_idx": collector_idx,
                    "sound_key":  defn.sound_key,
                }

        effect = ActiveEffect(
            definition    = defn,
            remaining     = defn.duration,
            collector_idx = collector_idx,
            target_idx    = target_idx,
        )
        self.stacks[target_idx].add(effect)

        # Call definition's on_collect hook
        try:
            defn.on_collect(collector_idx, players, puck, state)
        except Exception:
            logger.exception("on_collect error for %s", defn.id)

        logger.debug(
            "Activated %s: collector=%d target=%d duration=%.1fs",
            defn.id, collector_idx, target_idx, defn.duration,
        )
        return {
            "type":         "collected",
            "powerup_id":   defn.id,
            "player_idx":   collector_idx,
            "target_idx":   target_idx,
            "sound_key":    defn.sound_key,
            "particle_color": defn.particle_color,
            "position":     list(sphere.position),
        }

    # --- Per-frame effect ticks -------------------------------------------

    def _tick_active(
        self,
        dt: float,
        players: list,
        puck: Any,
        state: Any,
    ) -> None:
        for stack in self.stacks:
            for eff in stack.effects:
                if eff.definition.on_tick is not None:
                    try:
                        eff.definition.on_tick(dt, eff.target_idx, players, puck, state)
                    except Exception:
                        logger.exception("on_tick error for %s", eff.id)

    # --- Timer advancement -----------------------------------------------

    def _advance_timers(self, dt: float) -> None:
        """Subtract *dt* from every active effect's remaining time."""
        for stack in self.stacks:
            for eff in stack.effects:
                eff.tick(dt)

    # --- Expiry -----------------------------------------------------------

    def _do_expire_pass(
        self,
        players: list,
        puck: Any,
        state: Any,
    ) -> List[Dict]:
        """Scan both stacks for expired effects and fire on_expire callbacks."""
        events: List[Dict] = []
        for stack in self.stacks:
            to_remove = [eff for eff in stack.effects if eff.is_expired]
            for eff in to_remove:
                try:
                    eff.definition.on_expire(eff.target_idx, players, puck, state)
                except Exception:
                    logger.exception("on_expire error for %s", eff.id)
                stack.remove(eff)
                events.append({
                    "type":       "expired",
                    "powerup_id": eff.id,
                    "player_idx": eff.collector_idx,
                    "target_idx": eff.target_idx,
                    "sound_key":  "powerup_expire",
                })
                logger.debug("Expired %s for player %d", eff.id, eff.target_idx)
        return events

    # --- Speed application -----------------------------------------------

    def _apply_speed(self, players: list) -> None:
        """Push EffectStack speed multiplier to each player entity."""
        for i, player in enumerate(players):
            mult = self.stacks[i].get_speed_multiplier()
            if hasattr(player, "speed_multiplier"):
                player.speed_multiplier = mult

    # --- Paralysis enforcement -------------------------------------------

    def _apply_paralysis(self, players: list) -> None:
        """Freeze paralyzed players by zeroing their velocity intent."""
        for i, player in enumerate(players):
            paralyzed = self.stacks[i].is_paralyzed()
            if hasattr(player, "paralyzed"):
                player.paralyzed = paralyzed

    # --- Queries (for HUD, renderer, training adapter) -------------------

    def get_active_effects(self, player_idx: int) -> List[ActiveEffect]:
        """Return current active effects for *player_idx*."""
        return self.stacks[player_idx].effects

    def get_field_spheres(self) -> List[FieldSphere]:
        """Return field spheres for the renderer."""
        return list(self.field_spheres)

    # --- Reset -----------------------------------------------------------

    def reset(self) -> None:
        """Clear all state (call when a new match begins)."""
        for stack in self.stacks:
            stack.clear()
        self.field_spheres.clear()
        self._spawn_timer   = 0.0
        self._next_spawn_at = random.uniform(SPAWN_INTERVAL_MIN, SPAWN_INTERVAL_MAX)
        logger.debug("PowerUpManager reset")

    def __repr__(self) -> str:
        return (
            f"PowerUpManager("
            f"field={len(self.field_spheres)}, "
            f"stack0={self.stacks[0]}, "
            f"stack1={self.stacks[1]})"
        )
