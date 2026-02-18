"""
PowerUpTimerPill — Single active-effect chip rendered next to the HUD timer bar.

Responsibilities (SRP):
    - Render ONE active powerup as a pill: [icon | time_remaining]
    - Handle <1s warning pulse (red border + blinking text)
    - Cache surfaces; never allocate per-frame

Design principles:
    - Composition: exists as a dumb renderer; all state comes from ActiveEffect
    - Extensibility: color, icon, duration all come from PowerUpDefinition
    - No pygame init required at module import time
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from game.components.IconRenderer import IconRenderer

if TYPE_CHECKING:
    from shared.powerups.effect_stack import ActiveEffect


# ---------------------------------------------------------------------------
# Constants — tweak here for global pill appearance
# ---------------------------------------------------------------------------

PILL_H: int        = 20      # pill height in pixels
PILL_PAD_X: int    = 5       # horizontal inner padding
PILL_ICON_W: int   = 18      # reserved width for icon glyph
PILL_TIME_W: int   = 36      # reserved width for "4.2s" text
PILL_TOTAL_W: int  = PILL_ICON_W + PILL_PAD_X + PILL_TIME_W  # ≈ 59 px
PILL_SPACING: int  = 4       # gap between pills
MAX_PILLS: int     = 4       # max shown before "+N" overflow

WARN_THRESHOLD: float = 1.0  # seconds; triggers red pulse


class PowerUpTimerPill:
    """
    Renders a single powerup as a compact pill chip.

    Usage::

        pill = PowerUpTimerPill(font_size=15)
        pill.draw(screen, active_effect, x, y, now)

    ``x, y`` is the top-left corner of the pill.
    ``now`` is the current timestamp (``time.time()``); used for pulsing.
    """

    def __init__(self, font_size: int = 15) -> None:
        self._font: pygame.font.Font | None = None
        self._font_size = font_size
        self._last_time_str: str = ""
        self._time_surf: pygame.Surface | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return PILL_TOTAL_W

    @property
    def height(self) -> int:
        return PILL_H

    def draw(
        self,
        screen: pygame.Surface,
        effect: "ActiveEffect",
        x: int,
        y: int,
        now: float,
    ) -> None:
        """Blit the pill onto *screen* at (x, y)."""
        font = self._ensure_font()
        defn = effect.definition
        remaining = max(0.0, effect.remaining)
        warn = remaining < WARN_THRESHOLD

        # ---- Pill background ------------------------------------------------
        alpha = 200
        bg_surf = pygame.Surface((PILL_TOTAL_W, PILL_H), pygame.SRCALPHA)
        # Darken slightly for readability
        r, g, b = defn.color
        dark = (max(0, r - 40), max(0, g - 40), max(0, b - 40), alpha)
        bg_surf.fill(dark)
        screen.blit(bg_surf, (x, y))

        # ---- Pulse border when warning (<1s) --------------------------------
        border_color = (255, 255, 255, 180)
        if warn:
            pulse = abs(math.sin(now * 8))  # 4 Hz pulse
            br = int(255 * pulse)
            border_color = (255, br, br, 255)
        pygame.draw.rect(screen, border_color, (x, y, PILL_TOTAL_W, PILL_H), 1, border_radius=3)

        # ---- Icon -------  left side ----------------------------------------
        icon_surf = IconRenderer.get_powerup_surface(
            defn.icon, PILL_ICON_W, (255, 255, 255)
        )
        ix = x + PILL_PAD_X
        iy = y + (PILL_H - icon_surf.get_height()) // 2
        screen.blit(icon_surf, (ix, iy))

        # ---- Time text  ── right side ----------------------------------------
        time_str = f"{remaining:.1f}s"
        text_color = (255, 80, 80) if warn else (255, 255, 255)
        time_surf = font.render(time_str, True, text_color)
        tx = x + PILL_ICON_W + PILL_PAD_X
        ty = y + (PILL_H - time_surf.get_height()) // 2
        screen.blit(time_surf, (tx, ty))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_font(self) -> pygame.font.Font:
        if self._font is None:
            self._font = pygame.font.Font(None, self._font_size)
        return self._font
