"""
Air Hockey environment with power-ups for RL training.
Extends AirHockeyEnv with power-up spawning, collection, and effects.
The AI learns to collect power-ups and play with their effects.
"""
import math
import random
import numpy as np
import pygame
from gymnasium import spaces
from enum import IntEnum

from training.envs.base_env import AirHockeyEnv, ACTION_COUNT, MAX_STEPS, SCORE_LIMIT, DIAGONAL_FACTOR
from shared.config import GameConfig, COLORS


class PowerUpType(IntEnum):
    """Training-side power-up types (mirrors game/entities/powerups.py)."""
    SPEED_BOOST = 0      # +50% speed
    SIZE_INCREASE = 1    # +30% size
    SLOW_OPPONENT = 2    # -30% opponent speed
    PUCK_MAGNET = 3      # Attract puck
    SHIELD = 4           # Protect goal
    SHRINK_OPPONENT = 5  # -25% opponent size


POWERUP_DURATION = {
    PowerUpType.SPEED_BOOST: 5.0,
    PowerUpType.SIZE_INCREASE: 7.0,
    PowerUpType.SLOW_OPPONENT: 5.0,
    PowerUpType.PUCK_MAGNET: 4.0,
    PowerUpType.SHIELD: 3.0,
    PowerUpType.SHRINK_OPPONENT: 5.0,
}

POWERUP_MULTIPLIER = {
    PowerUpType.SPEED_BOOST: 1.5,
    PowerUpType.SIZE_INCREASE: 1.3,
    PowerUpType.SLOW_OPPONENT: 0.7,
    PowerUpType.PUCK_MAGNET: 1.0,
    PowerUpType.SHIELD: 1.0,
    PowerUpType.SHRINK_OPPONENT: 0.75,
}

NUM_POWERUP_TYPES = len(PowerUpType)
MAX_FIELD_POWERUPS = 2


class FieldPowerUp:
    """A power-up item on the field."""
    def __init__(self, ptype: PowerUpType, x: float, y: float, radius: float):
        self.type = ptype
        self.position = [x, y]
        self.radius = radius
        self.age = 0.0
        self.lifetime = 15.0  # seconds

    def is_expired(self) -> bool:
        return self.age >= self.lifetime

    def check_collision(self, entity_pos, entity_radius) -> bool:
        dist = math.hypot(self.position[0] - entity_pos[0],
                          self.position[1] - entity_pos[1])
        return dist <= self.radius + entity_radius


class ActiveEffect:
    """An active power-up effect on a player."""
    def __init__(self, ptype: PowerUpType, remaining: float, target: str):
        self.type = ptype
        self.remaining = remaining
        self.target = target  # "ai" or "human"


class AirHockeyWithPowerUpsEnv(AirHockeyEnv):
    """Extended env with power-ups. Observation space expanded to include power-up info.

    Additional observations (13 dims on top of base 13 = 26 total):
      [0]  powerup_0_exists (0/1)
      [1]  powerup_0_x (normalized)
      [2]  powerup_0_y (normalized)
      [3]  powerup_0_type (normalized 0-1)
      [4]  powerup_1_exists (0/1)
      [5]  powerup_1_x (normalized)
      [6]  powerup_1_y (normalized)
      [7]  powerup_1_type (normalized 0-1)
      [8]  ai_speed_boost_active (0/1)
      [9]  ai_size_increase_active (0/1)
      [10] ai_magnet_active (0/1)
      [11] ai_shield_active (0/1)
      [12] dist_to_nearest_powerup (normalized)
    """

    def __init__(self, render_mode=None, play_mode=False, config: GameConfig = None):
        # Initialize powerup state BEFORE super().__init__() because it calls reset()
        self._base_obs_size = 13
        self._powerup_obs_size = 13
        self.field_powerups: list[FieldPowerUp] = []
        self.active_effects: list[ActiveEffect] = []
        self.spawn_timer = 0.0
        self.next_spawn_time = random.uniform(10.0, 20.0)
        self.powerup_radius = 18.0
        self._ai_speed_mult = 1.0
        self._ai_size_mult = 1.0

        super().__init__(render_mode=render_mode, play_mode=play_mode, config=config)

        # Override observation space to 26 dimensions
        total_obs = self._base_obs_size + self._powerup_obs_size
        low = np.zeros(total_obs, dtype=np.float32)
        low[4] = -1.0  # puck vel_x
        low[5] = -1.0  # puck vel_y
        high = np.ones(total_obs, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.field_powerups = []
        self.active_effects = []
        self.spawn_timer = 0.0
        self.next_spawn_time = random.uniform(10.0, 20.0)
        self._ai_speed_mult = 1.0
        self._ai_size_mult = 1.0
        return self._get_observation(), info

    def step(self, action):
        # Adjust move_amount by AI speed multiplier before calling parent step
        original_move = self.physics.ai_move_amount
        self.physics.ai_move_amount = int(original_move * self._ai_speed_mult)

        obs, reward, terminated, truncated, info = super().step(action)

        # Restore original move amount
        self.physics.ai_move_amount = original_move

        # Simulate dt (~1/60s per step at 60fps training)
        dt = 1.0 / 60.0

        # Update spawn timer
        self.spawn_timer += dt
        if self.spawn_timer >= self.next_spawn_time and len(self.field_powerups) < MAX_FIELD_POWERUPS:
            self._spawn_random_powerup()
            self.spawn_timer = 0.0
            self.next_spawn_time = random.uniform(10.0, 20.0)

        # Expire old field power-ups
        for pu in self.field_powerups:
            pu.age += dt
        self.field_powerups = [pu for pu in self.field_powerups if not pu.is_expired()]

        # Check AI collection
        collected = []
        for pu in self.field_powerups:
            if pu.check_collision(self.ai_mallet_position, self.ai_mallet_radius):
                self._apply_powerup(pu, "ai")
                collected.append(pu)
                reward += 0.3  # Collection bonus
        for pu in collected:
            self.field_powerups.remove(pu)

        # Check human collection (opponent)
        collected_human = []
        for pu in self.field_powerups:
            if pu.check_collision(self.human_mallet.position, self.human_mallet.radius):
                self._apply_powerup(pu, "human")
                collected_human.append(pu)
        for pu in collected_human:
            self.field_powerups.remove(pu)

        # Update active effects
        expired = []
        for effect in self.active_effects:
            effect.remaining -= dt
            if effect.remaining <= 0:
                expired.append(effect)
            # Magnet effect
            if effect.type == PowerUpType.PUCK_MAGNET and effect.target == "ai":
                dx = self.ai_mallet_position[0] - self.puck.position[0]
                dy = self.ai_mallet_position[1] - self.puck.position[1]
                dist = math.hypot(dx, dy)
                if dist > 10:
                    force = 0.3
                    self.puck.velocity[0] += (dx / dist) * force
                    self.puck.velocity[1] += (dy / dist) * force
        for effect in expired:
            self._remove_effect(effect)
            self.active_effects.remove(effect)

        # Rebuild observation with power-up info
        obs = self._get_observation()
        info["powerups_collected"] = len(collected)
        info["active_effects"] = len(self.active_effects)

        return obs, reward, terminated, truncated, info

    def _spawn_random_powerup(self):
        W, H = self.config.width, self.config.height
        x = random.uniform(W * 0.25, W * 0.75)
        y = random.uniform(H * 0.15, H * 0.85)
        ptype = random.choice(list(PowerUpType))
        self.field_powerups.append(FieldPowerUp(ptype, x, y, self.powerup_radius))

    def _apply_powerup(self, pu: FieldPowerUp, target: str):
        duration = POWERUP_DURATION[pu.type]
        effect = ActiveEffect(pu.type, duration, target)
        self.active_effects.append(effect)

        if pu.type == PowerUpType.SPEED_BOOST:
            if target == "ai":
                self._ai_speed_mult = POWERUP_MULTIPLIER[pu.type]
        elif pu.type == PowerUpType.SLOW_OPPONENT:
            opposite = "human" if target == "ai" else "ai"
            eff = ActiveEffect(pu.type, duration, opposite)
            self.active_effects.append(eff)
            if opposite == "ai":
                self._ai_speed_mult = POWERUP_MULTIPLIER[pu.type]

    def _remove_effect(self, effect: ActiveEffect):
        if effect.type in (PowerUpType.SPEED_BOOST, PowerUpType.SLOW_OPPONENT):
            if effect.target == "ai":
                self._ai_speed_mult = 1.0

    def _get_observation(self):
        """Build 26-dim observation: 13 base + 13 power-up info."""
        base_obs = super()._get_observation()
        W, H = self.config.width, self.config.height

        pu_obs = np.zeros(self._powerup_obs_size, dtype=np.float32)

        # Field power-ups (up to 2)
        for i, pu in enumerate(self.field_powerups[:2]):
            offset = i * 4
            pu_obs[offset] = 1.0  # exists
            pu_obs[offset + 1] = pu.position[0] / W
            pu_obs[offset + 2] = pu.position[1] / H
            pu_obs[offset + 3] = pu.type / (NUM_POWERUP_TYPES - 1)

        # Active effects on AI
        for effect in self.active_effects:
            if effect.target == "ai":
                if effect.type == PowerUpType.SPEED_BOOST:
                    pu_obs[8] = 1.0
                elif effect.type == PowerUpType.SIZE_INCREASE:
                    pu_obs[9] = 1.0
                elif effect.type == PowerUpType.PUCK_MAGNET:
                    pu_obs[10] = 1.0
                elif effect.type == PowerUpType.SHIELD:
                    pu_obs[11] = 1.0

        # Distance to nearest power-up
        if self.field_powerups:
            min_dist = min(
                math.hypot(pu.position[0] - self.ai_mallet_position[0],
                           pu.position[1] - self.ai_mallet_position[1])
                for pu in self.field_powerups
            )
            pu_obs[12] = min(min_dist / math.hypot(W, H), 1.0)
        else:
            pu_obs[12] = 1.0  # No power-ups = max distance

        return np.concatenate([base_obs, pu_obs])
