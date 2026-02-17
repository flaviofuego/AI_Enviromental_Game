"""
Base Air Hockey Gymnasium environment for RL training.
Aligned with game parameters: dimensions, physics, 9-action space, score limit.
"""
import math
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from shared.config import GameConfig, PhysicsConfig, COLORS
from shared.entities.puck import Puck
from shared.entities.mallet import Mallet
from shared.entities.table import Table
from shared.physics import vector_length
from training.envs.rewards import RewardCalculator, FieldState
from training.envs.opponents import (
    AlgorithmicOpponent,
    FieldSnapshot,
    OpponentFactory,
    OpponentParams,
)
from training.envs.observation_builder import ObservationBuilder, ObsSnapshot

# 9 actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay,
#            5=UpLeft, 6=UpRight, 7=DownLeft, 8=DownRight
ACTION_COUNT = 9
MAX_STEPS = 2000
SCORE_LIMIT = 7
DIAGONAL_FACTOR = 0.7071  # 1/sqrt(2)


class AirHockeyEnv(gym.Env):
    """Custom Environment for Air Hockey that follows the Gymnasium interface.

    Aligned with game's GameConfig/PhysicsConfig for consistent behavior
    between training and gameplay.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        play_mode=False,
        config: GameConfig = None,
        obs_builder: ObservationBuilder | None = None,
    ):
        super().__init__()
        self.config = config or GameConfig()
        self.physics = PhysicsConfig()
        self.play_mode = play_mode

        self.action_space = spaces.Discrete(ACTION_COUNT)

        # Configurable observation builder (defaults to 13D for backward compat)
        self._obs_builder = obs_builder or ObservationBuilder.standard()
        self.observation_space = self._obs_builder.get_observation_space()

        self.steps_since_last_hit = 0
        self._edge_steps = 0
        self._last_hit_was_defensive = False
        self._consecutive_hits = 0
        self._puck_vel_pre_hit: list[float] | None = None

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.table = Table(self.config)
        self.puck = None
        self.ai_mallet = None
        self.human_mallet = None
        self.all_sprites = None

        self.player_score = 0
        self.ai_score = 0
        self.steps = 0
        self.max_steps = MAX_STEPS
        self.score_limit = SCORE_LIMIT
        self.opponent_skill = 0.3
        self.last_average_reward = -float('inf')

        # Componentized opponent system (Mejora 8)
        self.opponent = OpponentFactory.from_skill(self.opponent_skill)

        # Componentized reward system (Mejora 7)
        self.reward_calculator = RewardCalculator.default(gamma=0.995)
        self._last_reward_breakdown = None

        self.reset()

    def increase_opponent_difficulty(self, current_reward):
        if current_reward > self.last_average_reward + 0.5:
            self.opponent_skill = min(0.9, self.opponent_skill + 0.1)
            self.last_average_reward = current_reward
            # Update opponent parameters to match new skill level
            self.opponent.params = OpponentParams.from_skill(self.opponent_skill)
        return self.opponent_skill

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        W, H = self.config.width, self.config.height

        if not pygame.get_init():
            pygame.init()

        self.puck = Puck(self.config, self.physics)
        self.human_mallet = Mallet(W // 4, H // 2, COLORS.NEON_RED, self.config)

        # AI mallet — uses config.mallet_radius for consistency with game
        self.ai_mallet_radius = self.config.mallet_radius
        self.ai_mallet_position = [float(W * 3 // 4), float(H // 2)]
        self.ai_mallet_velocity = [0.0, 0.0]

        self.ai_mallet = pygame.sprite.Sprite()
        size = self.ai_mallet_radius * 2
        self.ai_mallet.image = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(self.ai_mallet.image, COLORS.NEON_GREEN,
                           (self.ai_mallet_radius, self.ai_mallet_radius), self.ai_mallet_radius)
        pygame.draw.circle(self.ai_mallet.image, (255, 255, 255, 150),
                           (self.ai_mallet_radius, self.ai_mallet_radius), self.ai_mallet_radius // 2)
        self.ai_mallet.rect = self.ai_mallet.image.get_rect(
            center=(int(self.ai_mallet_position[0]), int(self.ai_mallet_position[1])))
        self.ai_mallet.mask = pygame.mask.from_surface(self.ai_mallet.image)

        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.human_mallet, self.ai_mallet, self.puck)

        self.player_score = 0
        self.ai_score = 0
        self.steps = 0
        self.steps_since_last_hit = 0
        self._edge_steps = 0
        self._last_hit_was_defensive = False
        self._consecutive_hits = 0
        self._puck_vel_pre_hit = None

        # Reset reward components' internal state
        if hasattr(self, 'reward_calculator'):
            self.reward_calculator.reset()

        # Reset opponent episode state (randomize style bias)
        if hasattr(self, 'opponent'):
            self.opponent.reset_episode()

        if not self.play_mode:
            self._update_human_player()

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1
        W, H = self.config.width, self.config.height

        prev_position = self.ai_mallet_position.copy()
        move_amount = self.physics.ai_move_amount
        diag = move_amount * DIAGONAL_FACTOR
        radius = self.ai_mallet_radius

        # Apply action — 9 discrete actions including diagonals
        if action == 0:    # Up
            self.ai_mallet_position[1] = max(self.ai_mallet_position[1] - move_amount, radius)
        elif action == 1:  # Down
            self.ai_mallet_position[1] = min(self.ai_mallet_position[1] + move_amount, H - radius)
        elif action == 2:  # Left
            self.ai_mallet_position[0] = max(self.ai_mallet_position[0] - move_amount, W // 2 + radius)
        elif action == 3:  # Right
            self.ai_mallet_position[0] = min(self.ai_mallet_position[0] + move_amount, W - radius)
        elif action == 5:  # UpLeft
            self.ai_mallet_position[1] = max(self.ai_mallet_position[1] - diag, radius)
            self.ai_mallet_position[0] = max(self.ai_mallet_position[0] - diag, W // 2 + radius)
        elif action == 6:  # UpRight
            self.ai_mallet_position[1] = max(self.ai_mallet_position[1] - diag, radius)
            self.ai_mallet_position[0] = min(self.ai_mallet_position[0] + diag, W - radius)
        elif action == 7:  # DownLeft
            self.ai_mallet_position[1] = min(self.ai_mallet_position[1] + diag, H - radius)
            self.ai_mallet_position[0] = max(self.ai_mallet_position[0] - diag, W // 2 + radius)
        elif action == 8:  # DownRight
            self.ai_mallet_position[1] = min(self.ai_mallet_position[1] + diag, H - radius)
            self.ai_mallet_position[0] = min(self.ai_mallet_position[0] + diag, W - radius)
        # action == 4: Stay

        self.ai_mallet.rect.center = (int(self.ai_mallet_position[0]),
                                       int(self.ai_mallet_position[1]))
        self.ai_mallet_velocity = [
            self.ai_mallet_position[0] - prev_position[0],
            self.ai_mallet_position[1] - prev_position[1]
        ]

        if not self.play_mode:
            self._update_human_player()

        prev_distance = math.hypot(
            self.puck.position[0] - self.ai_mallet_position[0],
            self.puck.position[1] - self.ai_mallet_position[1]
        )

        # Capture puck velocity BEFORE collision for shot direction analysis
        self._puck_vel_pre_hit = self.puck.velocity.copy()

        self.puck.update()

        ai_hit_puck = self._check_mallet_collision(
            self.ai_mallet, self.ai_mallet_position,
            self.ai_mallet_radius, self.ai_mallet_velocity
        )
        human_hit_puck = self.puck.check_mallet_collision(self.human_mallet)

        goal = self.table.is_goal(self.puck)
        goal_scored = False

        if goal == "player":
            self.player_score += 1
            goal_scored = True
            self.puck.reset("player")
        elif goal == "ai":
            self.ai_score += 1
            goal_scored = True
            self.puck.reset("ai")

        reward = self._calculate_reward(prev_distance, ai_hit_puck, goal)
        self._update_hit_tracking(ai_hit_puck)
        terminated = goal_scored or self.player_score >= self.score_limit or self.ai_score >= self.score_limit
        truncated = self.steps >= self.max_steps and not terminated

        observation = self._get_observation()
        info = {
            "player_score": self.player_score,
            "ai_score": self.ai_score,
            "steps": self.steps,
            "hit_puck": ai_hit_puck
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _build_field_snapshot(self) -> FieldSnapshot:
        """Build an immutable field snapshot for the opponent system."""
        W, H = self.config.width, self.config.height
        goal_width = H * self.physics.goal_width_ratio
        goal_y1 = H * (1 - self.physics.goal_width_ratio) / 2
        goal_y2 = H * (1 + self.physics.goal_width_ratio) / 2
        return FieldSnapshot(
            width=W,
            height=H,
            half_width=W // 2,
            mallet_pos=(self.human_mallet.position[0], self.human_mallet.position[1]),
            mallet_radius=self.human_mallet.radius,
            puck_pos=(self.puck.position[0], self.puck.position[1]),
            puck_vel=(self.puck.velocity[0], self.puck.velocity[1]),
            puck_radius=self.puck.radius,
            own_goal_x=0.0,
            own_goal_center_y=H / 2,
            goal_y1=goal_y1,
            goal_y2=goal_y2,
            rival_goal_x=float(W),
            rival_goal_center_y=H / 2,
        )

    def _update_human_player(self):
        """Simulated opponent for training — delegates to AlgorithmicOpponent."""
        snapshot = self._build_field_snapshot()
        current_pos = (self.human_mallet.position[0], self.human_mallet.position[1])

        new_x, new_y, vx, vy = self.opponent.update(snapshot, current_pos)

        self.human_mallet.position[0] = new_x
        self.human_mallet.position[1] = new_y
        self.human_mallet.rect.center = (int(new_x), int(new_y))
        self.human_mallet.velocity = [vx, vy]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self):
        """Build observation via the configurable ObservationBuilder."""
        snap = ObsSnapshot(
            width=self.config.width,
            height=self.config.height,
            ai_pos=(self.ai_mallet_position[0], self.ai_mallet_position[1]),
            ai_vel=(self.ai_mallet_velocity[0], self.ai_mallet_velocity[1]),
            puck_pos=(self.puck.position[0], self.puck.position[1]),
            puck_vel=(self.puck.velocity[0], self.puck.velocity[1]),
            puck_max_speed=self.puck.max_speed,
            opponent_pos=(self.human_mallet.position[0], self.human_mallet.position[1]),
            opponent_vel=(
                self.human_mallet.velocity[0] if hasattr(self.human_mallet, 'velocity') else 0.0,
                self.human_mallet.velocity[1] if hasattr(self.human_mallet, 'velocity') else 0.0,
            ),
            steps_since_last_hit=self.steps_since_last_hit,
            player_score=self.player_score,
            ai_score=self.ai_score,
            score_limit=self.score_limit,
        )
        return self._obs_builder.build(snap)

    # ------------------------------------------------------------------
    # Reward — Componentized reward system (Mejora 7)
    # ------------------------------------------------------------------

    def _calculate_reward(self, prev_distance, ai_hit_puck, goal):
        """Delegate to componentized RewardCalculator.

        Builds a FieldState snapshot and passes it to all reward components.
        Stores the breakdown for debugging/logging.
        """
        W, H = self.config.width, self.config.height
        puck_speed = math.hypot(self.puck.velocity[0], self.puck.velocity[1])
        current_distance = math.hypot(
            self.puck.position[0] - self.ai_mallet_position[0],
            self.puck.position[1] - self.ai_mallet_position[1]
        )

        state = FieldState(
            width=W,
            height=H,
            ai_pos=(self.ai_mallet_position[0], self.ai_mallet_position[1]),
            ai_vel=(self.ai_mallet_velocity[0], self.ai_mallet_velocity[1]),
            puck_pos=(self.puck.position[0], self.puck.position[1]),
            puck_vel=(self.puck.velocity[0], self.puck.velocity[1]),
            puck_speed=puck_speed,
            puck_max_speed=self.puck.max_speed,
            puck_vel_pre_hit=(
                (self._puck_vel_pre_hit[0], self._puck_vel_pre_hit[1])
                if self._puck_vel_pre_hit is not None else None
            ),
            ai_puck_distance=current_distance,
            ai_puck_prev_distance=prev_distance,
            ai_mallet_radius=self.ai_mallet_radius,
            ai_hit_puck=ai_hit_puck,
            goal=goal,
            puck_in_ai_half=self.puck.position[0] > W / 2,
            puck_heading_toward_ai=self.puck.velocity[0] > 0,
            player_score=self.player_score,
            ai_score=self.ai_score,
            score_limit=self.score_limit,
            steps_since_last_hit=self.steps_since_last_hit,
            consecutive_hits=self._consecutive_hits,
        )

        reward, breakdown = self.reward_calculator.calculate(state)
        self._last_reward_breakdown = breakdown
        return reward

    def _update_hit_tracking(self, ai_hit_puck: bool) -> None:
        """Update hit counters after reward calculation."""
        if ai_hit_puck:
            self._consecutive_hits += 1
            self.steps_since_last_hit = 0
        else:
            self.steps_since_last_hit += 1
            if self.steps_since_last_hit > 5:
                self._consecutive_hits = 0

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def _check_mallet_collision(self, mallet, position, radius, velocity):
        dx = self.puck.position[0] - position[0]
        dy = self.puck.position[1] - position[1]
        distance = math.hypot(dx, dy)

        if distance <= self.puck.radius + radius:
            if distance > 0:
                dx /= distance
                dy /= distance
            else:
                dx, dy = 1.0, 0.0

            W, H = self.config.width, self.config.height
            self.puck.position[0] = position[0] + (radius + self.puck.radius + 1) * dx
            self.puck.position[1] = position[1] + (radius + self.puck.radius + 1) * dy
            self.puck.position[0] = max(self.puck.radius, min(self.puck.position[0], W - self.puck.radius))
            self.puck.position[1] = max(self.puck.radius, min(self.puck.position[1], H - self.puck.radius))

            speed_contrib = math.hypot(velocity[0], velocity[1]) * 0.5
            self.puck.velocity[0] = dx * (6 + speed_contrib)
            self.puck.velocity[1] = dy * (6 + speed_contrib)
            self.puck.rect.center = (int(self.puck.position[0]), int(self.puck.position[1]))
            return True
        return False

    def render(self):
        if self.render_mode is None:
            return
        W, H = self.config.width, self.config.height

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((W, H))
                pygame.display.set_caption("Air Hockey - Training")
            else:
                self.screen = pygame.Surface((W, H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return False

        self.table.draw(self.screen)
        self.all_sprites.draw(self.screen)
        font = pygame.font.Font(None, 36)
        txt = font.render(f"{self.player_score} - {self.ai_score}", True, COLORS.WHITE)
        self.screen.blit(txt, (W // 2 - txt.get_width() // 2, 20))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        if self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        return True

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
