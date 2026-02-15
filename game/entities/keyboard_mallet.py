"""Keyboard-controlled mallet for Player vs Player local mode."""
import pygame
from shared.config import COLORS, GameConfig
from shared.entities.mallet import Mallet


class KeyboardMallet(Mallet):
    """
    Mallet controlled by keyboard (WASD or arrow keys), restricted to the right half.
    Used for Player 2 in local PvP mode.
    """

    def __init__(self, config: GameConfig = None, color=COLORS.NEON_GREEN,
                 custom_image=None, use_arrows: bool = True):
        cfg = config or GameConfig()
        super().__init__(cfg.width * 3 // 4, cfg.height // 2, color, cfg, custom_image)
        self.use_arrows = use_arrows
        self.base_speed = 8.0  # Calibrated to be competitive vs mouse

    def update(self, keys=None):
        """Update position based on keyboard state."""
        if keys is None:
            keys = pygame.key.get_pressed()

        cfg = self.config
        self.prev_position = self.position.copy()
        speed = self.base_speed * self.speed_multiplier

        dx, dy = 0.0, 0.0
        if self.use_arrows:
            if keys[pygame.K_UP]:
                dy -= speed
            if keys[pygame.K_DOWN]:
                dy += speed
            if keys[pygame.K_LEFT]:
                dx -= speed
            if keys[pygame.K_RIGHT]:
                dx += speed
        else:  # WASD
            if keys[pygame.K_w]:
                dy -= speed
            if keys[pygame.K_s]:
                dy += speed
            if keys[pygame.K_a]:
                dx -= speed
            if keys[pygame.K_d]:
                dx += speed

        # Normalize diagonal movement
        if dx != 0 and dy != 0:
            factor = 0.7071  # 1/sqrt(2)
            dx *= factor
            dy *= factor

        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        # Restrict to right half
        new_x = max(cfg.half_width + self.radius, min(new_x, cfg.width - self.radius))
        new_y = max(self.radius, min(new_y, cfg.height - self.radius))

        self.position = [new_x, new_y]
        self.rect.center = (int(self.position[0]), int(self.position[1]))
        self.velocity = [
            self.position[0] - self.prev_position[0],
            self.position[1] - self.prev_position[1],
        ]
