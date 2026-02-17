"""Pause menu overlay."""
import pygame
from shared.config import COLORS
from game.components.FontCache import font_cache


class PauseMenu:
    """Draws pause overlay. Can be extended with menu options."""

    def __init__(self, config):
        self.config = config
        # Cache fonts once
        self._font = font_cache.get(None, 48)
        self._hint_font = font_cache.get(None, 24)
        # Cache overlay surface
        self._overlay = pygame.Surface(
            (config.width, config.height), pygame.SRCALPHA
        )
        self._overlay.fill((0, 0, 0, 120))

    def draw(self, screen):
        W, H = self.config.width, self.config.height
        screen.blit(self._overlay, (0, 0))

        txt = self._font.render("PAUSA", True, COLORS.WHITE)
        screen.blit(txt, (W // 2 - txt.get_width() // 2, H // 2 - txt.get_height() // 2))

        hint = self._hint_font.render("Presiona ESC para continuar", True, (180, 180, 180))
        screen.blit(hint, (W // 2 - hint.get_width() // 2, H // 2 + 40))
