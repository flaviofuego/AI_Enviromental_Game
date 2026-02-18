"""
PowerUpOpponentBadge — Compact "negative-effect" indicator shown on the opponent side.

Responsibilities (SRP):
    - Render effects that this player *applied to* the opponent (slow, paralyze)
      from the opponent's perspective with a distinct red-bordered style.
    - Visual language: dark pill + red border = "debuff on opponent"

Design:
    - Inherits the same pill geometry as PowerUpTimerPill for visual consistency
    - Separate component so it can evolve independently (different icon, layout)
    - Purely presentational

Usage::

    badge = PowerUpOpponentBadge(font_size=15)
    badge.draw(screen, effects_affecting_opponent, anchor_x, anchor_y, now)
"""
from __future__ import annotations

import math
from typing import List, TYPE_CHECKING

import pygame

from game.components.IconRenderer import IconRenderer
from game.components.PowerUpTimerPill import PILL_H, PILL_TOTAL_W, PILL_SPACING, MAX_PILLS

if TYPE_CHECKING:
    from shared.powerups.effect_stack import ActiveEffect


_BADGE_BG_ALPHA  = 180
_BORDER_NEGATIVE = (255, 80, 80)   # red → communicates "applied to opponent"
_WARN_THRESHOLD  = 1.0             # seconds


class PowerUpOpponentBadge:
    """
    Row of compact red-bordered chips showing debuffs the local player
    has *cast onto* the opponent.

    The chips appear on the opponent's side of the HUD so both players
    can see what debuffs are in play without looking at their own half.

    Parameters
    ----------
    align : "left" | "right"
        Same convention as PowerUpTimerBar.
    font_size : int
    """

    def __init__(self, align: str = "right", font_size: int = 14) -> None:
        self._align     = align
        self._font: pygame.font.Font | None = None
        self._font_size = font_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        screen: pygame.Surface,
        effects: "List[ActiveEffect]",
        anchor_x: int,
        anchor_y: int,
        now: float,
    ) -> None:
        if not effects:
            return

        font = self._ensure_font()
        visible = effects[:MAX_PILLS]
        step = PILL_TOTAL_W + PILL_SPACING

        if self._align == "left":
            x = anchor_x
        else:
            total = len(visible) * step - PILL_SPACING
            x = anchor_x - total

        for eff in visible:
            self._draw_badge(screen, font, eff, x, anchor_y, now)
            x += step

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _draw_badge(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        effect: "ActiveEffect",
        x: int,
        y: int,
        now: float,
    ) -> None:
        remaining = max(0.0, effect.remaining)
        warn = remaining < _WARN_THRESHOLD

        # Dark background tinted slightly in the deflt color
        r, g, b = effect.definition.color
        tint = (max(0, r - 80), max(0, g - 80), max(0, b - 80), _BADGE_BG_ALPHA)
        bg = pygame.Surface((PILL_TOTAL_W, PILL_H), pygame.SRCALPHA)
        bg.fill(tint)
        screen.blit(bg, (x, y))

        # Red border (pulse faster during warn)
        if warn:
            pulse = abs(math.sin(now * 10))
            br = int(255 * pulse)
            border = (255, br, br, 255)
        else:
            border = _BORDER_NEGATIVE
        pygame.draw.rect(screen, border, (x, y, PILL_TOTAL_W, PILL_H), 1, border_radius=3)

        # Icon
        icon_surf = IconRenderer.get_powerup_surface(
            effect.definition.icon, 14, (255, 180, 180)
        )
        screen.blit(icon_surf, (x + 3, y + (PILL_H - icon_surf.get_height()) // 2))

        # Time
        time_str = f"{remaining:.1f}s"
        t_color = (255, 80, 80) if warn else (255, 160, 160)
        t_surf = font.render(time_str, True, t_color)
        tx = x + 18 + 3
        screen.blit(t_surf, (tx, y + (PILL_H - t_surf.get_height()) // 2))

    def _ensure_font(self) -> pygame.font.Font:
        if self._font is None:
            self._font = pygame.font.Font(None, self._font_size)
        return self._font
