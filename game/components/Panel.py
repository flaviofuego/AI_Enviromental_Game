"""
Reusable Panel component for drawing semi-transparent panels with
title, border, and content area.
"""
import pygame
import math


class Panel:
    """A styled, semi-transparent panel that can contain arbitrary content."""

    def __init__(self, x: int, y: int, width: int, height: int,
                 bg_color=(20, 20, 40, 200),
                 border_color=(173, 216, 230),
                 border_width: int = 2,
                 border_radius: int = 0,
                 title: str = "",
                 title_font: pygame.font.Font = None,
                 title_color=(173, 216, 230)):
        self.rect = pygame.Rect(x, y, width, height)
        self.bg_color = bg_color
        self.border_color = border_color
        self.border_width = border_width
        self.border_radius = border_radius
        self.title = title
        self.title_font = title_font or pygame.font.Font(None, 24)
        self.title_color = title_color

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def x(self):
        return self.rect.x

    @property
    def y(self):
        return self.rect.y

    @property
    def width(self):
        return self.rect.width

    @property
    def height(self):
        return self.rect.height

    @property
    def content_rect(self) -> pygame.Rect:
        """Rectangle available for content (below title if present)."""
        top_offset = 0
        if self.title:
            top_offset = self.title_font.get_linesize() + 16
        return pygame.Rect(
            self.rect.x + 10,
            self.rect.y + top_offset + 10,
            self.rect.width - 20,
            self.rect.height - top_offset - 20,
        )

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface, glow: bool = False,
             animation_time: float = 0):
        """Draw the panel onto *surface*."""
        # Background
        panel_surf = pygame.Surface(
            (self.rect.width, self.rect.height), pygame.SRCALPHA
        )
        panel_surf.fill(self.bg_color)
        surface.blit(panel_surf, self.rect.topleft)

        # Border with optional glow
        if glow:
            intensity = abs(math.sin(animation_time * 3)) * 0.3 + 0.7
            color = tuple(
                min(255, int(c * intensity))
                for c in self.border_color[:3]
            )
        else:
            color = self.border_color

        pygame.draw.rect(
            surface, color, self.rect,
            self.border_width, border_radius=self.border_radius,
        )

        # Title
        if self.title:
            title_surf = self.title_font.render(
                self.title, True, self.title_color
            )
            title_rect = title_surf.get_rect(
                centerx=self.rect.centerx, top=self.rect.y + 10
            )
            surface.blit(title_surf, title_rect)

    def collidepoint(self, pos) -> bool:
        return self.rect.collidepoint(pos)
