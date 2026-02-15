"""Pause menu overlay."""
import pygame
from shared.config import COLORS


class PauseMenu:
    """Draws pause overlay. Can be extended with menu options."""

    def __init__(self, config):
        self.config = config

    def draw(self, screen):
        W, H = self.config.width, self.config.height
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        screen.blit(overlay, (0, 0))

        font = pygame.font.Font(None, 48)
        txt = font.render("PAUSA", True, COLORS.WHITE)
        screen.blit(txt, (W // 2 - txt.get_width() // 2, H // 2 - txt.get_height() // 2))

        hint_font = pygame.font.Font(None, 24)
        hint = hint_font.render("Presiona ESC para continuar", True, (180, 180, 180))
        screen.blit(hint, (W // 2 - hint.get_width() // 2, H // 2 + 40))
