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

    # ------------------------------------------------------------------
    # PowerUp icons — drawn entirely with pygame primitives
    # ------------------------------------------------------------------

    @staticmethod
    def draw_powerup_speed(surface: pygame.Surface, rect: pygame.Rect,
                           color: tuple = (255, 255, 255)) -> None:
        """Lightning bolt — speed_boost."""
        cx, cy = rect.centerx, rect.centery
        h = min(rect.width, rect.height) // 2 - 2
        w = int(h * 0.55)
        # Bolt: two triangles forming a zigzag
        top    = [(cx + w // 2, cy - h),   (cx - w // 4, cy),    (cx + w // 3, cy)]
        bottom = [(cx + w // 4, cy),        (cx - w // 2, cy + h), (cx - w // 3, cy)]
        pygame.draw.polygon(surface, color, top)
        pygame.draw.polygon(surface, color, bottom)

    @staticmethod
    def draw_powerup_magnet(surface: pygame.Surface, rect: pygame.Rect,
                            color: tuple = (255, 255, 255)) -> None:
        """U-shaped magnet — magnet."""
        cx, cy = rect.centerx, rect.centery
        r = min(rect.width, rect.height) // 2 - 3
        arm_w = max(3, r // 3)
        # Horseshoe arc (top half)
        pygame.draw.arc(surface, color,
                        (cx - r, cy - r, r * 2, r * 2),
                        0, math.pi, arm_w)
        # Left arm (downward)
        pygame.draw.line(surface, color, (cx - r, cy), (cx - r, cy + r // 2), arm_w)
        # Right arm (downward)
        pygame.draw.line(surface, color, (cx + r, cy), (cx + r, cy + r // 2), arm_w)
        # Left tip (colored end)
        pygame.draw.line(surface, (255, 80, 80),
                         (cx - r - arm_w // 2, cy + r // 2),
                         (cx - r + arm_w // 2 + 1, cy + r // 2), arm_w + 1)
        # Right tip
        pygame.draw.line(surface, (80, 80, 255),
                         (cx + r - arm_w // 2, cy + r // 2),
                         (cx + r + arm_w // 2 + 1, cy + r // 2), arm_w + 1)

    @staticmethod
    def draw_powerup_shield(surface: pygame.Surface, rect: pygame.Rect,
                            color: tuple = (255, 255, 255)) -> None:
        """Shield outline — shield."""
        cx, cy = rect.centerx, rect.centery
        h = min(rect.width, rect.height) // 2 - 2
        w = int(h * 0.75)
        # Shield polygon: flat top, rounded bottom point
        pts = [
            (cx - w, cy - h),
            (cx + w, cy - h),
            (cx + w, cy),
            (cx,     cy + h),
            (cx - w, cy),
        ]
        pygame.draw.polygon(surface, color, pts, 2)
        # Inner cross
        pygame.draw.line(surface, color, (cx, cy - h + 3), (cx, cy + h - 4), 1)

    @staticmethod
    def draw_powerup_slow(surface: pygame.Surface, rect: pygame.Rect,
                          color: tuple = (255, 255, 255)) -> None:
        """Snowflake — slow_opponent."""
        cx, cy = rect.centerx, rect.centery
        r = min(rect.width, rect.height) // 2 - 2
        # Six arms
        for i in range(6):
            angle = math.radians(i * 60)
            ex = cx + int(r * math.cos(angle))
            ey = cy + int(r * math.sin(angle))
            pygame.draw.line(surface, color, (cx, cy), (ex, ey), 2)
            # Small branches at 60 % length
            br = int(r * 0.45)
            bx = cx + int(br * math.cos(angle))
            by = cy + int(br * math.sin(angle))
            for sign in (+1, -1):
                branch_a = angle + sign * math.radians(60)
                bex = bx + int((r * 0.3) * math.cos(branch_a))
                bey = by + int((r * 0.3) * math.sin(branch_a))
                pygame.draw.line(surface, color, (bx, by), (bex, bey), 1)

    @staticmethod
    def draw_powerup_paralyze(surface: pygame.Surface, rect: pygame.Rect,
                               color: tuple = (255, 255, 255)) -> None:
        """Double lightning bolt — paralyze."""
        cx, cy = rect.centerx, rect.centery
        h = min(rect.width, rect.height) // 2 - 2
        w = max(3, int(h * 0.35))
        offsets = [-w, w]
        for ox in offsets:
            top    = [(cx + ox + w // 2, cy - h),
                      (cx + ox - w // 4, cy),
                      (cx + ox + w // 3, cy)]
            bottom = [(cx + ox + w // 4, cy),
                      (cx + ox - w // 2, cy + h),
                      (cx + ox - w // 3, cy)]
            pygame.draw.polygon(surface, color, top)
            pygame.draw.polygon(surface, color, bottom)

    @staticmethod
    def draw_powerup_invisibility(surface: pygame.Surface, rect: pygame.Rect,
                                   color: tuple = (255, 255, 255)) -> None:
        """Eye with dashed iris — invisibility."""
        cx, cy = rect.centerx, rect.centery
        rx = min(rect.width, rect.height) // 2 - 2
        ry = max(3, rx // 2)
        # Outer eye outline (arc top + arc bottom)
        pygame.draw.arc(surface, color,
                        (cx - rx, cy - ry, rx * 2, ry * 2),
                        0, math.pi, 2)
        pygame.draw.arc(surface, color,
                        (cx - rx, cy - ry, rx * 2, ry * 2),
                        math.pi, math.tau, 2)
        # Pupil circle (dashed via two arcs)
        pr = max(2, rx // 3)
        pygame.draw.circle(surface, color, (cx, cy), pr, 2)
        # Strike-through — "hidden"
        pygame.draw.line(surface, color,
                         (cx - rx, cy - ry), (cx + rx, cy + ry), 2)

    @staticmethod
    def draw_powerup_duplication(surface: pygame.Surface, rect: pygame.Rect,
                                  color: tuple = (255, 255, 255)) -> None:
        """Two overlapping circles — duplication."""
        cx, cy = rect.centerx, rect.centery
        r = min(rect.width, rect.height) // 2 - 4
        offset = max(3, r // 2)
        pygame.draw.circle(surface, color, (cx - offset // 2, cy), r, 2)
        pygame.draw.circle(surface, color, (cx + offset // 2, cy), r, 2)

    @staticmethod
    def draw_powerup_obstacle(surface: pygame.Surface, rect: pygame.Rect,
                               color: tuple = (255, 255, 255)) -> None:
        """Diamond ice block — obstacle."""
        cx, cy = rect.centerx, rect.centery
        h = min(rect.width, rect.height) // 2 - 2
        pts = [
            (cx,     cy - h),
            (cx + h, cy),
            (cx,     cy + h),
            (cx - h, cy),
        ]
        pygame.draw.polygon(surface, color, pts, 2)
        # Inner cross lines for ice texture
        mid = h // 2
        pygame.draw.line(surface, color, (cx - mid, cy - mid), (cx + mid, cy + mid), 1)
        pygame.draw.line(surface, color, (cx + mid, cy - mid), (cx - mid, cy + mid), 1)

    # ------------------------------------------------------------------
    # PowerUp icon helper — draw by powerup id into an SRCALPHA surface
    # ------------------------------------------------------------------

    #: Maps powerup id → draw method name in this class
    POWERUP_ICON_MAP: dict = {
        "speed_boost":  "powerup_speed",
        "magnet":       "powerup_magnet",
        "shield":       "powerup_shield",
        "slow_opponent":"powerup_slow",
        "paralyze":     "powerup_paralyze",
        "invisibility": "powerup_invisibility",
        "duplication":  "powerup_duplication",
        "obstacle":     "powerup_obstacle",
    }

    @classmethod
    def get_powerup_surface(
        cls,
        powerup_id: str,
        size: int,
        color: tuple = (255, 255, 255),
        alpha: int = 255,
    ) -> pygame.Surface:
        """
        Return a cached SRCALPHA surface with the icon for *powerup_id*.

        Falls back to a plain filled circle when the id is unknown.
        Color and alpha are applied to the drawn primitives.
        """
        key = ("pu", powerup_id, size, color, alpha)
        if key not in cls._icon_cache:
            surf = cls._create_surface(size)
            rect = pygame.Rect(0, 0, size, size)
            icon_name = cls.POWERUP_ICON_MAP.get(powerup_id)
            draw_fn = getattr(cls, f"draw_{icon_name}", None) if icon_name else None
            if draw_fn:
                draw_fn(surf, rect, color)
            else:
                # Fallback: filled circle
                r = size // 2 - 2
                pygame.draw.circle(surf, color, (size // 2, size // 2), r, 2)
            if alpha < 255:
                surf.set_alpha(alpha)
            cls._icon_cache[key] = surf
        cached = cls._icon_cache[key]
        if alpha < 255:
            # Return a copy so callers can set_alpha without polluting cache
            c = cached.copy()
            c.set_alpha(alpha)
            return c
        return cached
