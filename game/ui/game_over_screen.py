"""Game over screen with stats, environmental info, and action buttons."""
import pygame
from shared.config import COLORS


class GameOverScreen:
    """Draws the game over overlay with match statistics."""

    def __init__(self, config):
        self.config = config
        self._fonts = {}

    def _font(self, size):
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    def draw(self, screen, state, level_config):
        """Draw game over overlay. Called from GameEngine._draw_game_over."""
        # The main game_engine handles this directly for now.
        # This class can be expanded for richer game over screens.
        pass
