"""
Icon renderer using pygame primitives.
Replaces emoji/unicode glyphs with resolution-independent drawn icons.
All icons are rendered onto cached surfaces for reuse.
"""
import pygame
import math
from functools import lru_cache


class IconRenderer:
    """Draws simple icons using pygame primitives. No emoji, no external assets."""

    @staticmethod
    def _create_surface(size: int) -> pygame.Surface:
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        return surf

    # ------------------------------------------------------------------
    # Arrow icons
    # ------------------------------------------------------------------

    @staticmethod
    def draw_arrow_left(surface: pygame.Surface, rect: pygame.Rect,
                        color: tuple = (255, 255, 255), padding: int = 8):
        """Draw a left-pointing triangle arrow inside *rect*."""
        cx, cy = rect.centerx, rect.centery
        half = min(rect.width, rect.height) // 2 - padding
        points = [
            (cx - half, cy),
            (cx + half, cy - half),
            (cx + half, cy + half),
        ]
        pygame.draw.polygon(surface, color, points)

    @staticmethod
    def draw_arrow_right(surface: pygame.Surface, rect: pygame.Rect,
                         color: tuple = (255, 255, 255), padding: int = 8):
        """Draw a right-pointing triangle arrow inside *rect*."""
        cx, cy = rect.centerx, rect.centery
        half = min(rect.width, rect.height) // 2 - padding
        points = [
            (cx + half, cy),
            (cx - half, cy - half),
            (cx - half, cy + half),
        ]
        pygame.draw.polygon(surface, color, points)

    @staticmethod
    def draw_arrow_up(surface: pygame.Surface, rect: pygame.Rect,
                      color: tuple = (255, 255, 255), padding: int = 8):
        """Draw an upward-pointing triangle arrow inside *rect*."""
        cx, cy = rect.centerx, rect.centery
        half = min(rect.width, rect.height) // 2 - padding
        points = [
            (cx, cy - half),
            (cx - half, cy + half),
            (cx + half, cy + half),
        ]
        pygame.draw.polygon(surface, color, points)

    @staticmethod
    def draw_arrow_down(surface: pygame.Surface, rect: pygame.Rect,
                        color: tuple = (255, 255, 255), padding: int = 8):
        """Draw a downward-pointing triangle arrow inside *rect*."""
        cx, cy = rect.centerx, rect.centery
        half = min(rect.width, rect.height) // 2 - padding
        points = [
            (cx, cy + half),
            (cx - half, cy - half),
            (cx + half, cy - half),
        ]
        pygame.draw.polygon(surface, color, points)

    # ------------------------------------------------------------------
    # Status / popup type icons
    # ------------------------------------------------------------------

    @staticmethod
    def draw_info(surface: pygame.Surface, rect: pygame.Rect,
                  color: tuple = (100, 150, 200)):
        """Draw an (i) info circle icon."""
        cx, cy = rect.centerx, rect.centery
        r = min(rect.width, rect.height) // 2 - 2
        pygame.draw.circle(surface, color, (cx, cy), r, 2)
        # Dot
        pygame.draw.circle(surface, color, (cx, cy - r // 3), 2)
        # Stem
        pygame.draw.line(surface, color,
                         (cx, cy - r // 6), (cx, cy + r // 2), 2)

    @staticmethod
    def draw_warning(surface: pygame.Surface, rect: pygame.Rect,
                     color: tuple = (255, 200, 50)):
        """Draw a triangle-exclamation warning icon."""
        cx, cy = rect.centerx, rect.centery
        r = min(rect.width, rect.height) // 2 - 2
        # Triangle
        points = [
            (cx, cy - r),
            (cx - r, cy + r),
            (cx + r, cy + r),
        ]
        pygame.draw.polygon(surface, color, points, 2)
        # Exclamation
        pygame.draw.line(surface, color,
                         (cx, cy - r // 3), (cx, cy + r // 3), 2)
        pygame.draw.circle(surface, color, (cx, cy + r * 2 // 3), 2)

    @staticmethod
    def draw_error(surface: pygame.Surface, rect: pygame.Rect,
                   color: tuple = (220, 100, 100)):
        """Draw an X-in-circle error icon."""
        cx, cy = rect.centerx, rect.centery
        r = min(rect.width, rect.height) // 2 - 2
        pygame.draw.circle(surface, color, (cx, cy), r, 2)
        offset = int(r * 0.55)
        pygame.draw.line(surface, color,
                         (cx - offset, cy - offset),
                         (cx + offset, cy + offset), 2)
        pygame.draw.line(surface, color,
                         (cx + offset, cy - offset),
                         (cx - offset, cy + offset), 2)

    @staticmethod
    def draw_help(surface: pygame.Surface, rect: pygame.Rect,
                  color: tuple = (100, 200, 150)):
        """Draw a question-mark-in-circle help icon."""
        cx, cy = rect.centerx, rect.centery
        r = min(rect.width, rect.height) // 2 - 2
        pygame.draw.circle(surface, color, (cx, cy), r, 2)
        # Question mark curve approximation
        arc_r = r // 3
        pygame.draw.arc(surface, color,
                        (cx - arc_r, cy - r + 4, arc_r * 2, arc_r * 2),
                        math.radians(-30), math.radians(200), 2)
        pygame.draw.line(surface, color,
                         (cx, cy - 2), (cx, cy + r // 6), 2)
        pygame.draw.circle(surface, color, (cx, cy + r // 3 + 2), 2)

    @staticmethod
    def draw_checkmark(surface: pygame.Surface, rect: pygame.Rect,
                       color: tuple = (100, 220, 100)):
        """Draw a checkmark icon."""
        cx, cy = rect.centerx, rect.centery
        r = min(rect.width, rect.height) // 2 - 4
        points = [
            (cx - r, cy),
            (cx - r // 3, cy + r),
            (cx + r, cy - r),
        ]
        pygame.draw.lines(surface, color, False, points, 3)

    @staticmethod
    def draw_close(surface: pygame.Surface, rect: pygame.Rect,
                   color: tuple = (220, 100, 100)):
        """Draw an X close icon."""
        pad = min(rect.width, rect.height) // 4
        pygame.draw.line(surface, color,
                         (rect.left + pad, rect.top + pad),
                         (rect.right - pad, rect.bottom - pad), 3)
        pygame.draw.line(surface, color,
                         (rect.right - pad, rect.top + pad),
                         (rect.left + pad, rect.bottom - pad), 3)

    @staticmethod
    def draw_play(surface: pygame.Surface, rect: pygame.Rect,
                  color: tuple = (255, 255, 255), padding: int = 6):
        """Draw a play (right-pointing triangle) icon."""
        IconRenderer.draw_arrow_right(surface, rect, color, padding)

    # ------------------------------------------------------------------
    # Cached icon surfaces
    # ------------------------------------------------------------------

    _icon_cache: dict[tuple, pygame.Surface] = {}

    @classmethod
    def get_icon_surface(cls, icon_name: str, size: int,
                         color: tuple = (255, 255, 255)) -> pygame.Surface:
        """Return a cached surface with the given icon drawn on it."""
        key = (icon_name, size, color)
        if key not in cls._icon_cache:
            surf = cls._create_surface(size)
            rect = pygame.Rect(0, 0, size, size)
            draw_fn = getattr(cls, f"draw_{icon_name}", None)
            if draw_fn:
                draw_fn(surf, rect, color)
            cls._icon_cache[key] = surf
        return cls._icon_cache[key]

    # ------------------------------------------------------------------
    # Popup-type icon mapping
    # ------------------------------------------------------------------

    POPUP_ICON_MAP = {
        "info": ("info", (100, 150, 200)),
        "warning": ("warning", (255, 200, 50)),
        "error": ("error", (220, 100, 100)),
        "help": ("help", (100, 200, 150)),
    }

    @classmethod
    def get_popup_icon(cls, popup_type: str, size: int = 24) -> pygame.Surface:
        """Return the icon surface appropriate for a popup type."""
        name, color = cls.POPUP_ICON_MAP.get(popup_type, ("info", (100, 150, 200)))
        return cls.get_icon_surface(name, size, color)
