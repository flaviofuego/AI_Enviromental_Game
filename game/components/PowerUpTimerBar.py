"""
PowerUpTimerBar — Renders a horizontal strip of PowerUpTimerPills for one player.

Responsibilities (SRP):
    - Layout N pills in a row (left-to-right for player, right-to-left for opponent)
    - Show "+N" overflow chip when more than MAX_PILLS are active
    - Handle empty state (renders nothing)

Design:
    - Receives a list[ActiveEffect] — no coupling to EffectStack internals
    - Purely presentational; does NOT mutate any state
    - Fully composable: instantiate two bars (one per player) independently

Usage::

    bar = PowerUpTimerBar(align="left", font_size=15)
    bar.draw(screen, effects, anchor_x, anchor_y, now)
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

import pygame

from game.components.PowerUpTimerPill import (
    PowerUpTimerPill,
    PILL_H,
    PILL_TOTAL_W,
    PILL_SPACING,
    MAX_PILLS,
)

if TYPE_CHECKING:
    from shared.powerups.effect_stack import ActiveEffect


class PowerUpTimerBar:
    """
    Strips of pills representing a single player's active powerups.

    Parameters
    ----------
    align : "left" | "right"
        "left"  → pills grow rightward from anchor_x (player 1 / self).
        "right" → pills grow leftward from anchor_x  (player 2 / opponent).
    font_size : int
        Font size forwarded to every PowerUpTimerPill.
    """

    _OVERFLOW_BG  = (60, 60, 60, 200)
    _OVERFLOW_FG  = (220, 220, 220)

    def __init__(self, align: str = "left", font_size: int = 15) -> None:
        if align not in ("left", "right"):
            raise ValueError("align must be 'left' or 'right'")
        self._align = align
        self._pill  = PowerUpTimerPill(font_size=font_size)
        self._font: pygame.font.Font | None = None
        self._font_size = font_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def height(self) -> int:
        return PILL_H

    def total_width(self, effect_count: int) -> int:
        """Compute total pixel width for a given number of effects."""
        n = min(effect_count, MAX_PILLS)
        if effect_count > MAX_PILLS:
            n = MAX_PILLS  # pills
            return n * (PILL_TOTAL_W + PILL_SPACING) + PILL_TOTAL_W  # +overflow chip
        return max(0, n * PILL_TOTAL_W + max(0, n - 1) * PILL_SPACING)

    def draw(
        self,
        screen: pygame.Surface,
        effects: "List[ActiveEffect]",
        anchor_x: int,
        anchor_y: int,
        now: float,
    ) -> None:
        """
        Blit all pills onto *screen*.

        anchor_x / anchor_y: starting corner.
          - align="left"  → anchor is LEFT edge of first pill
          - align="right" → anchor is RIGHT edge of last pill
        """
        if not effects:
            return

        visible = effects[:MAX_PILLS]
        overflow = len(effects) - MAX_PILLS

        # Build left→right list of (effect | None for overflow chip)
        items: list = list(visible)
        if overflow > 0:
            items.append(None)  # sentinel for overflow chip

        step = PILL_TOTAL_W + PILL_SPACING

        if self._align == "left":
            x = anchor_x
            for item in items:
                if item is None:
                    self._draw_overflow(screen, overflow, x, anchor_y)
                else:
                    self._pill.draw(screen, item, x, anchor_y, now)
                x += step
        else:
            # Right-aligned: last pill's right edge = anchor_x
            total = len(items) * step - PILL_SPACING
            x = anchor_x - total
            for item in items:
                if item is None:
                    self._draw_overflow(screen, overflow, x, anchor_y)
                else:
                    self._pill.draw(screen, item, x, anchor_y, now)
                x += step

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _draw_overflow(
        self, screen: pygame.Surface, count: int, x: int, y: int
    ) -> None:
        """Render a "+N" chip indicating hidden effects."""
        font = self._ensure_font()
        label = f"+{count}"
        bg = pygame.Surface((PILL_TOTAL_W, PILL_H), pygame.SRCALPHA)
        bg.fill(self._OVERFLOW_BG)
        screen.blit(bg, (x, y))
        pygame.draw.rect(screen, (120, 120, 120), (x, y, PILL_TOTAL_W, PILL_H), 1, border_radius=3)
        surf = font.render(label, True, self._OVERFLOW_FG)
        sx = x + (PILL_TOTAL_W - surf.get_width()) // 2
        sy = y + (PILL_H - surf.get_height()) // 2
        screen.blit(surf, (sx, sy))

    def _ensure_font(self) -> pygame.font.Font:
        if self._font is None:
            self._font = pygame.font.Font(None, self._font_size)
        return self._font
