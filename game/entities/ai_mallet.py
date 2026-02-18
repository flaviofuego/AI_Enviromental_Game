"""AI-controlled mallet: simple heuristic or RL model integration."""
import random
from shared.config import COLORS, GameConfig
from shared.entities.mallet import Mallet


class AIMallet(Mallet):
    """Mallet controlled by simple AI heuristics. RL control is handled externally."""

    def __init__(self, config: GameConfig = GameConfig(), custom_image=None,
                 reaction_speed=None, prediction_factor=None):
        cfg = config
        super().__init__(cfg.width * 3 // 4, cfg.height // 2, COLORS.NEON_GREEN, cfg, custom_image)
        self.reaction_speed = reaction_speed if reaction_speed is not None else 0.1
        self.prediction_factor = prediction_factor if prediction_factor is not None else 0.4
        self.defensive_position = [cfg.width * 3 // 4, cfg.height // 2]

    def update_simple_ai(self, puck_pos):
        """Simple tracking AI behavior (used when RL model is not available)."""
        # Paralysis: freeze mallet
        if self.paralyzed:
            self.velocity = [0.0, 0.0]
            return

        cfg = self.config
        self.prev_position = self.position.copy()

        if puck_pos and puck_pos[0] > cfg.half_width:
            is_near_corner = (
                (puck_pos[0] < self.radius * 3 or puck_pos[0] > cfg.width - self.radius * 3) and
                (puck_pos[1] < self.radius * 3 or puck_pos[1] > cfg.height - self.radius * 3)
            )
            if is_near_corner:
                target_x = cfg.width * 3 // 4
                target_y = cfg.height // 2
                self.position[0] += (target_x - self.position[0]) * 0.15
                self.position[1] += (target_y - self.position[1]) * 0.15
            else:
                ix = min(max(puck_pos[0] + random.randint(-20, 20),
                             cfg.half_width + self.radius), cfg.width - self.radius)
                iy = min(max(puck_pos[1] + random.randint(-20, 20),
                             self.radius), cfg.height - self.radius)
                self.position[0] += (ix - self.position[0]) * self.reaction_speed
                self.position[1] += (iy - self.position[1]) * self.reaction_speed

            self.rect.center = (int(self.position[0]), int(self.position[1]))
            self.velocity = [
                self.position[0] - self.prev_position[0],
                self.position[1] - self.prev_position[1],
            ]

    def apply_rl_action(self, action: int, move_amount: int = 7):
        """Apply a discrete RL action to this mallet.
        Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay,
                 5=UpLeft, 6=UpRight, 7=DownLeft, 8=DownRight
        """
        # Paralysis: force Stay regardless of chosen action
        if self.paralyzed:
            self.velocity = [0.0, 0.0]
            return

        cfg = self.config
        self.prev_position = self.position.copy()
        diag = move_amount * 0.7071  # 1/sqrt(2) for diagonal normalization

        if action == 0:    # Up
            self.position[1] = max(self.position[1] - move_amount, self.radius)
        elif action == 1:  # Down
            self.position[1] = min(self.position[1] + move_amount, cfg.height - self.radius)
        elif action == 2:  # Left
            self.position[0] = max(self.position[0] - move_amount, cfg.half_width + self.radius)
        elif action == 3:  # Right
            self.position[0] = min(self.position[0] + move_amount, cfg.width - self.radius)
        elif action == 5:  # UpLeft
            self.position[1] = max(self.position[1] - diag, self.radius)
            self.position[0] = max(self.position[0] - diag, cfg.half_width + self.radius)
        elif action == 6:  # UpRight
            self.position[1] = max(self.position[1] - diag, self.radius)
            self.position[0] = min(self.position[0] + diag, cfg.width - self.radius)
        elif action == 7:  # DownLeft
            self.position[1] = min(self.position[1] + diag, cfg.height - self.radius)
            self.position[0] = max(self.position[0] - diag, cfg.half_width + self.radius)
        elif action == 8:  # DownRight
            self.position[1] = min(self.position[1] + diag, cfg.height - self.radius)
            self.position[0] = min(self.position[0] + diag, cfg.width - self.radius)
        # action == 4: Stay

        self.rect.center = (int(self.position[0]), int(self.position[1]))
        self.velocity = [
            self.position[0] - self.prev_position[0],
            self.position[1] - self.prev_position[1],
        ]
