"""
TimerDisplay: reusable HUD component that renders a match timer.

Single Responsibility: rendering only — receives elapsed/remaining time,
outputs pixels.  No time-tracking logic.

Performance:
- Font cached on init (never created per-frame).
- Text surface re-rendered only when the displayed string changes.
- Background pill pre-rendered and reused.
- ``convert_alpha()`` called once at cache time.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import pygame


class TimerMode(Enum):
    """How the timer should display."""
    COUNTUP = auto()    # Show elapsed time  (↑ MM:SS)
    COUNTDOWN = auto()  # Show remaining     (MM:SS)


class TimerDisplay:
    """Lightweight, cacheable timer renderer.

    Parameters
    ----------
    font_size : int
        Base font size for the timer digits.
    mode : TimerMode
        Whether to show elapsed (COUNTUP) or remaining (COUNTDOWN).
    time_limit : float | None
        Total seconds for countdown mode; ignored in countup.
    warning_threshold : float
        Seconds remaining below which the timer blinks red (countdown only).
    position : tuple[int, int] | None
        Explicit ``(x, y)`` centre position.  If *None*, defaults to
        horizontally-centred at ``y_offset`` from the top.
    y_offset : int
        Vertical offset from the top when *position* is None.
    """

    # Colours (static, avoid per-instance allocation)
    _COLOR_NORMAL: tuple[int, int, int] = (255, 255, 255)
    _COLOR_WARNING: tuple[int, int, int] = (255, 80, 80)
    _COLOR_WARNING_ALT: tuple[int, int, int] = (255, 200, 200)
    _COLOR_BG: tuple[int, int, int, int] = (0, 0, 0, 100)
    _COLOR_BORDER: tuple[int, int, int, int] = (255, 255, 255, 60)

    def __init__(
        self,
        font_size: int = 26,
        mode: TimerMode = TimerMode.COUNTUP,
        time_limit: Optional[float] = None,
        warning_threshold: float = 30.0,
        position: Optional[tuple[int, int]] = None,
        y_offset: int = 50,
    ) -> None:
        self._font_size = font_size
        self._mode = mode
        self._time_limit = time_limit
        self._warning_threshold = warning_threshold
        self._position = position
        self._y_offset = y_offset

        # Font (created once)
        self._font: pygame.font.Font = pygame.font.Font(None, font_size)

        # Cache state — avoid re-rendering identical text
        self._cached_text: str = ""
        self._cached_color: tuple[int, int, int] = self._COLOR_NORMAL
        self._cached_surface: Optional[pygame.Surface] = None
        self._cached_bg: Optional[pygame.Surface] = None

        # Prefix icon surfaces (created once)
        self._icon_up: Optional[pygame.Surface] = None
        self._icon_down: Optional[pygame.Surface] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(
        self,
        mode: Optional[TimerMode] = None,
        time_limit: Optional[float] = None,
    ) -> None:
        """Reconfigure mode/limit at runtime (e.g. when match rules change)."""
        if mode is not None:
            self._mode = mode
        if time_limit is not None:
            self._time_limit = time_limit
        # Invalidate cache so next draw picks up changes
        self._cached_text = ""

    def draw(self, surface: pygame.Surface, elapsed: float) -> None:
        """Render the timer onto *surface*.

        Parameters
        ----------
        surface : pygame.Surface
            Target surface (usually the screen).
        elapsed : float
            Seconds elapsed since match start (already pause-compensated).
        """
        display_seconds, color = self._resolve_display(elapsed)
        text = self._format_time(display_seconds)

        # Re-render text surface only when content or colour changed
        if text != self._cached_text or color != self._cached_color:
            self._cached_text = text
            self._cached_color = color
            self._cached_surface = self._font.render(text, True, color)
            if self._cached_surface.get_flags() & pygame.SRCALPHA == 0:
                self._cached_surface = self._cached_surface.convert_alpha()
            # Rebuild background pill to match text width
            self._rebuild_bg(self._cached_surface)

        if self._cached_surface is None:
            return

        # --- Positioning ---
        sw = surface.get_width()
        txt_w = self._cached_surface.get_width()
        txt_h = self._cached_surface.get_height()

        # Icon (▲ / ▼) rendered once
        icon = self._get_icon(color)
        icon_w = icon.get_width() + 4 if icon else 0
        total_w = icon_w + txt_w

        if self._position is not None:
            cx, cy = self._position
        else:
            cx = sw // 2
            cy = self._y_offset

        x = cx - total_w // 2
        y = cy

        # Draw background pill
        if self._cached_bg is not None:
            bg_x = x - 8
            bg_y = y - 3
            surface.blit(self._cached_bg, (bg_x, bg_y))

        # Draw icon
        if icon:
            surface.blit(icon, (x, y + (txt_h - icon.get_height()) // 2))
            x += icon_w

        # Draw text
        surface.blit(self._cached_surface, (x, y))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_display(self, elapsed: float) -> tuple[float, tuple[int, int, int]]:
        """Return (seconds_to_display, color) based on mode."""
        if self._mode == TimerMode.COUNTDOWN and self._time_limit and self._time_limit > 0:
            remaining = max(0.0, self._time_limit - elapsed)
            if remaining < self._warning_threshold:
                # Blink between warning colours at ~2 Hz
                blink = int(remaining * 2) % 2 == 0
                color = self._COLOR_WARNING if blink else self._COLOR_WARNING_ALT
            else:
                color = self._COLOR_NORMAL
            return remaining, color

        # COUNTUP (default fallback)
        return elapsed, self._COLOR_NORMAL

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        total = int(seconds)
        m = total // 60
        s = total % 60
        return f"{m:02d}:{s:02d}"

    def _rebuild_bg(self, text_surface: pygame.Surface) -> None:
        """Pre-render a translucent pill behind the timer text."""
        icon = self._get_icon(self._cached_color)
        icon_w = (icon.get_width() + 4) if icon else 0
        w = text_surface.get_width() + icon_w + 16
        h = text_surface.get_height() + 6
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill(self._COLOR_BG)
        pygame.draw.rect(bg, self._COLOR_BORDER, bg.get_rect(), 1, border_radius=6)
        self._cached_bg = bg

    def _get_icon(self, color: tuple[int, int, int]) -> Optional[pygame.Surface]:
        """Return a small ▲ (countup) or ▼ (countdown) indicator surface."""
        if self._mode == TimerMode.COUNTUP:
            if self._icon_up is None or self._icon_color_mismatch(self._icon_up, color):
                self._icon_up = self._render_icon("▲", color)
            return self._icon_up
        elif self._mode == TimerMode.COUNTDOWN:
            if self._icon_down is None or self._icon_color_mismatch(self._icon_down, color):
                self._icon_down = self._render_icon("▼", color)
            return self._icon_down
        return None

    def _render_icon(self, char: str, color: tuple[int, int, int]) -> pygame.Surface:
        small_font = pygame.font.Font(None, max(14, self._font_size - 8))
        surf = small_font.render(char, True, color)
        return surf.convert_alpha()

    @staticmethod
    def _icon_color_mismatch(surf: pygame.Surface, color: tuple[int, int, int]) -> bool:
        """Check if the existing icon needs re-rendering for a new colour."""
        # Quick heuristic: sample centre pixel
        try:
            cx, cy = surf.get_width() // 2, surf.get_height() // 2
            px = surf.get_at((cx, cy))
            return (px.r, px.g, px.b) != color[:3]
        except Exception:
            return True
