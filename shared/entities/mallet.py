"""
Base Mallet sprite used by both game and training.
Subclassed by HumanMallet, AIMallet, KeyboardMallet in game/entities/.
"""
import pygame
from shared.config import GameConfig, COLORS


class Mallet(pygame.sprite.Sprite):
    """Base class for all hockey mallets."""

    def __init__(self, x: float, y: float, color=COLORS.NEON_RED,
                 config: GameConfig = None, custom_image=None):
        super().__init__()
        self.config = config or GameConfig()
        self.radius = self.config.mallet_radius

        # Visual
        if custom_image is not None:
            self.image = custom_image
            if self.image.get_size() != (self.radius * 2, self.radius * 2):
                self.image = pygame.transform.smoothscale(
                    self.image, (self.radius * 2, self.radius * 2)
                )
        else:
            self.image = pygame.Surface(
                (self.radius * 2, self.radius * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                self.image, color, (self.radius, self.radius), self.radius
            )
            pygame.draw.circle(
                self.image, (255, 255, 255, 150),
                (self.radius, self.radius), self.radius // 2,
            )

        self.rect = self.image.get_rect(center=(int(x), int(y)))
        self.mask = self._create_circular_mask()

        # Physics state
        self.position = [float(x), float(y)]
        self.prev_position = self.position.copy()
        self.velocity = [0.0, 0.0]

        # Power-up modifiers (1.0 = normal)
        self.speed_multiplier: float = 1.0
        self._size_modifier: float = 1.0
        self._base_radius: int = self.radius

    # ------------------------------------------------------------------
    # Collision mask
    # ------------------------------------------------------------------

    def _create_circular_mask(self) -> pygame.mask.Mask:
        mask_surf = pygame.Surface(
            (self.radius * 2, self.radius * 2), pygame.SRCALPHA
        )
        pygame.draw.circle(
            mask_surf, (255, 255, 255, 255), (self.radius, self.radius), self.radius
        )
        return pygame.mask.from_surface(mask_surf)

    # ------------------------------------------------------------------
    # Power-up helpers
    # ------------------------------------------------------------------

    def apply_size_modifier(self, multiplier: float):
        """
        Resize the mallet temporarily.
        Pass 1.0 to reset to the original size.
        """
        self._size_modifier = multiplier
        self.radius = max(8, int(self._base_radius * multiplier))
        # Regenerate image and mask at new size
        color = COLORS.NEON_RED  # fallback
        self.image = pygame.Surface(
            (self.radius * 2, self.radius * 2), pygame.SRCALPHA
        )
        pygame.draw.circle(
            self.image, color, (self.radius, self.radius), self.radius
        )
        pygame.draw.circle(
            self.image, (255, 255, 255, 150),
            (self.radius, self.radius), self.radius // 2,
        )
        self.rect = self.image.get_rect(center=(int(self.position[0]), int(self.position[1])))
        self.mask = self._create_circular_mask()
