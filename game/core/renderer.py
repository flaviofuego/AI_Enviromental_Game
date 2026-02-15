"""
Optimized renderer using dirty rects, font caching, and pre-rendered static surfaces.
"""
import pygame
from shared.config import GameConfig, COLORS


class Renderer:
    """Handles all drawing operations with performance optimizations."""

    def __init__(self, screen: pygame.Surface, config: GameConfig):
        self.screen = screen
        self.config = config
        # Font cache
        self._fonts = {}
        # Pre-rendered static background (table lines, border)
        self._static_bg = None
        self._bg_dirty = True

    def get_font(self, size: int) -> pygame.font.Font:
        """Get or create a cached font of the given size."""
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    def pre_render_background(self, table, theme_bg=None):
        """
        Pre-render the static background (table lines, goals, border) to a surface.
        Call once or when the window size changes.
        """
        self._static_bg = pygame.Surface((self.config.width, self.config.height))
        if theme_bg is not None:
            self._static_bg.blit(theme_bg, (0, 0))
        else:
            self._static_bg.fill(table.table_color)
        # Draw static table elements onto the cached surface
        table.draw(self._static_bg, draw_background=not (theme_bg is not None))
        self._bg_dirty = False

    def draw_background(self):
        """Blit the pre-rendered background."""
        if self._static_bg is not None:
            self.screen.blit(self._static_bg, (0, 0))
        else:
            self.screen.fill(COLORS.BLACK)

    def invalidate_background(self):
        """Mark background as needing re-render."""
        self._bg_dirty = True

    @property
    def needs_bg_update(self) -> bool:
        return self._bg_dirty or self._static_bg is None
