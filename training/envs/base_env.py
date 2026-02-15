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

    def __init__(self, render_mode=None, play_mode=False, config: GameConfig = None):
        super().__init__()
        self.config = config or GameConfig()
        self.physics = PhysicsConfig()
        self.play_mode = play_mode

        self.action_space = spaces.Discrete(ACTION_COUNT)

        # 13-dimensional observation (normalized [0,1] or [-1,1])
        self.steps_since_last_hit = 0
        self._edge_steps = 0
        self._last_hit_was_defensive = False
        self._consecutive_hits = 0
        low = np.array([0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

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

        self.reset()

    def increase_opponent_difficulty(self, current_reward):
        if current_reward > self.last_average_reward + 0.5:
            self.opponent_skill = min(0.9, self.opponent_skill + 0.1)
            self.last_average_reward = current_reward
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

    def _update_human_player(self):
        """Simulated human opponent for training."""
        W, H = self.config.width, self.config.height
        prediction_ability = 0.3 + (0.7 * self.opponent_skill)
        reaction_speed = 0.05 + (0.2 * self.opponent_skill)
        accuracy = 0.5 + (0.5 * self.opponent_skill)
        aggression = 0.3 + (0.6 * self.opponent_skill)

        if self.puck.position[0] < W // 2:
            predicted_y = self.puck.position[1]
            if abs(self.puck.velocity[1]) > 0.5:
                time_to_intercept = (self.human_mallet.position[0] - self.puck.position[0]) / max(1.0, abs(self.puck.velocity[0]))
                perfect_prediction = self.puck.position[1] + self.puck.velocity[1] * time_to_intercept
                error = (1.0 - prediction_ability) * np.random.normal(0, H * 0.2)
                predicted_y = perfect_prediction + error

            dist = np.sqrt((self.puck.position[0] - self.human_mallet.position[0]) ** 2 +
                           (self.puck.position[1] - self.human_mallet.position[1]) ** 2)

            if dist < 150 * aggression:
                target_x = min(self.puck.position[0], W // 2 - self.human_mallet.radius)
                target_y = predicted_y
                if dist < 50:
                    angle = math.atan2(H / 2 - self.puck.position[1], W - self.puck.position[0])
                    angle_error = (1.0 - accuracy) * np.random.normal(0, 0.5)
                    target_y = self.puck.position[1] + 30 * math.sin(angle + angle_error)
            else:
                target_x = W * 0.25
                target_y = H / 2
                if self.puck.position[1] < H * 0.3:
                    target_y = H * 0.3
                elif self.puck.position[1] > H * 0.7:
                    target_y = H * 0.7
        else:
            traj = self.puck.velocity[1] / max(0.1, abs(self.puck.velocity[0]))
            potential_y = self.puck.position[1] + traj * (W // 2 - self.puck.position[0])
            target_x = W * 0.15
            target_y = np.clip(potential_y, H * 0.2, H * 0.8) + np.random.normal(0, H * 0.05)

        max_speed = 15.0
        move_x = (target_x - self.human_mallet.position[0]) * reaction_speed
        move_y = (target_y - self.human_mallet.position[1]) * reaction_speed
        mag = math.hypot(move_x, move_y)
        if mag > max_speed:
            scale = max_speed / mag
            move_x *= scale
            move_y *= scale

        self.human_mallet.position[0] = np.clip(
            self.human_mallet.position[0] + move_x,
            self.human_mallet.radius, W // 2 - self.human_mallet.radius
        )
        self.human_mallet.position[1] = np.clip(
            self.human_mallet.position[1] + move_y,
            self.human_mallet.radius, H - self.human_mallet.radius
        )
        self.human_mallet.rect.center = (int(self.human_mallet.position[0]),
                                          int(self.human_mallet.position[1]))
        self.human_mallet.velocity = [move_x, move_y]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self):
        W, H = self.config.width, self.config.height
        basic = np.array([
            self.ai_mallet_position[0] / W,
            self.ai_mallet_position[1] / H,
            self.puck.position[0] / W,
            self.puck.position[1] / H,
            np.clip(self.puck.velocity[0] / self.puck.max_speed, -1, 1),
            np.clip(self.puck.velocity[1] / self.puck.max_speed, -1, 1),
        ], dtype=np.float32)

        dist = math.hypot(
            self.puck.position[0] - self.ai_mallet_position[0],
            self.puck.position[1] - self.ai_mallet_position[1]
        ) / math.hypot(W, H)

        return np.append(basic, [
            dist,
            (W - self.puck.position[0]) / W,
            self.puck.position[0] / W,
            min(self.steps_since_last_hit / 100.0, 1.0),
            1.0 if self.puck.velocity[0] < 0 else 0.0,
            self.player_score / float(self.score_limit),
            self.ai_score / float(self.score_limit),
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Reward — Hockey-inspired reward shaping
    # ------------------------------------------------------------------

    def _calculate_reward(self, prev_distance, ai_hit_puck, goal):
        """
        Reward function inspired by real ice hockey concepts:
        - Gap control: maintain optimal distance to puck
        - Positional play: stay between puck and own goal
        - Interception: reward defensive saves
        - Shot quality: alignment x speed toward opponent goal
        - Counterattack: quick transition from defense to offense
        - Net-front discipline: don't camp at own goal
        """
        W, H = self.config.width, self.config.height
        reward = 0.0
        current_distance = math.hypot(
            self.puck.position[0] - self.ai_mallet_position[0],
            self.puck.position[1] - self.ai_mallet_position[1]
        )
        puck_in_ai_half = self.puck.position[0] > W / 2
        puck_speed = math.hypot(self.puck.velocity[0], self.puck.velocity[1])

        # ---- GAP CONTROL ----
        # Optimal distance ~100px: close enough to react, far enough to not get beaten
        if puck_in_ai_half:
            optimal_dist = 100.0
            gap_error = abs(current_distance - optimal_dist) / optimal_dist
            gap_reward = max(0, 1.0 - gap_error) * 0.05
            reward += gap_reward
            # Reward approaching when too far
            if current_distance > optimal_dist * 1.5 and current_distance < prev_distance:
                reward += 0.08 * min(1.0, (prev_distance - current_distance) * 0.1)

        # ---- POSITIONAL PLAY: between puck and own goal ----
        # AI's goal is at x=W. Good position = ai_x between puck_x and W
        if puck_in_ai_half:
            if self.ai_mallet_position[0] > self.puck.position[0]:
                coverage = 1.0 - abs(self.ai_mallet_position[0] - (self.puck.position[0] + W) / 2) / (W / 2)
                reward += 0.03 * max(0, coverage)

        # ---- Y-AXIS ALIGNMENT (centering / angle coverage) ----
        if puck_in_ai_half:
            y_diff = abs(self.ai_mallet_position[1] - self.puck.position[1])
            y_alignment = 1.0 - min(y_diff / (H * 0.5), 1.0)
            reward += 0.02 * y_alignment

        # ---- HIT PUCK: directional shot quality ----
        if ai_hit_puck:
            reward += 0.8  # Base hit reward

            # Shot quality: alignment with opponent's goal (at x=0, y=H/2)
            goal_center = [0.0, H / 2.0]
            gv = [goal_center[0] - self.puck.position[0],
                  goal_center[1] - self.puck.position[1]]
            gl = math.hypot(gv[0], gv[1])
            if gl > 0:
                gv = [gv[0] / gl, gv[1] / gl]

            pvn = [0.0, 0.0]
            if puck_speed > 0:
                pvn = [self.puck.velocity[0] / puck_speed,
                       self.puck.velocity[1] / puck_speed]
            alignment = gv[0] * pvn[0] + gv[1] * pvn[1]
            speed_ratio = min(1.0, puck_speed / self.puck.max_speed)

            # Shot on goal: positive alignment = puck heading toward opponent goal
            if alignment > 0:
                shot_quality = alignment * speed_ratio
                reward += 2.5 * shot_quality
                # Hard shot bonus
                if speed_ratio > 0.6:
                    reward += 0.5

            # ---- INTERCEPTION (defensive save) ----
            puck_was_incoming = self.puck.velocity[0] > 2.0 or (
                prev_distance < 80 and self.ai_mallet_position[0] > W * 0.7)
            if puck_was_incoming:
                reward += 1.0
                self._last_hit_was_defensive = True
            else:
                self._last_hit_was_defensive = False

            # ---- COUNTERATTACK ----
            if self._last_hit_was_defensive and self.puck.velocity[0] < -1.0:
                reward += 0.5
                self._last_hit_was_defensive = False

            self._consecutive_hits += 1
            self.steps_since_last_hit = 0
        else:
            self.steps_since_last_hit += 1
            if self.steps_since_last_hit > 5:
                self._consecutive_hits = 0

        # ---- GOAL REWARDS (asymmetric: favor offense) ----
        if goal == "player":
            reward -= 3.0  # Goal conceded
        elif goal == "ai":
            reward += 5.0  # Goal scored

        # ---- PRESSURE PLAY ----
        if not puck_in_ai_half:
            dist_to_center = abs(self.ai_mallet_position[0] - (W * 0.6))
            center_proximity = 1.0 - min(dist_to_center / (W * 0.3), 1.0)
            reward += 0.01 * center_proximity

        # ---- NET-FRONT DISCIPLINE ----
        dist_to_own_goal = W - self.ai_mallet_position[0]
        if dist_to_own_goal < self.ai_mallet_radius * 2:
            reward -= 0.05

        # ---- INACTIVITY PENALTY ----
        movement = math.hypot(self.ai_mallet_velocity[0], self.ai_mallet_velocity[1])
        if movement < 0.1 and puck_in_ai_half and current_distance < 200:
            reward -= 0.02

        return reward

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
