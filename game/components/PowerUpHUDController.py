"""
PowerUpHUDController — Orchestrator for all powerup-related HUD elements.

Responsibilities:
    - Decide WHAT to show and WHERE (layout logic lives here, not in hud.py)
    - Partition active effects into: self-buffs vs opponent-debuffs
    - Forward rendering to PowerUpTimerBar (buffs) and PowerUpOpponentBadge (debuffs)
    - Expose a single draw() entry-point consumed by HUD

Layout strategy
───────────────
The HUD bar is:

  [ Player score  2 - 1  AI score  |  timer  ]

Powerup pills attach *outside* the central pill so the score/timer block
is never obscured.  Both strips grow away from the centre:

  [P1 buffs ←→ ][  score | timer  ][ ←→ P2 buffs ]
         ↑ debuffs on opponent side ↑

When no powerups are active, no pixel is drawn → zero visual pollution.

Design:
    - Single Responsibility: only handles layout + dispatch
    - Open/Closed: adding a new player or a different renderer only
      requires subclassing or injecting an alternative bar
    - Does NOT know the score / timer values — only the geometry

Usage::

    ctrl = PowerUpHUDController()
    # inside HUD.draw():
    ctrl.draw(screen, powerup_manager, score_bar_rect, now)
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

import pygame

from game.components.PowerUpTimerBar import PowerUpTimerBar
from game.components.PowerUpOpponentBadge import PowerUpOpponentBadge
from game.components.PowerUpTimerPill import PILL_H, PILL_SPACING

if TYPE_CHECKING:
    from shared.powerups.manager import PowerUpManager
    from shared.powerups.effect_stack import ActiveEffect


# ---------------------------------------------------------------------------
# Vertical offset below the score bar (px)
# ---------------------------------------------------------------------------
_BAR_GAP: int = 4      # gap between score bar bottom and pill strip top


class PowerUpHUDController:
    """
    Coordinates all powerup HUD widgets.

    One instance lives inside HUD; draw() is called every frame.

    Parameters
    ----------
    font_size : int
        Forwarded to all child components for consistent text size.
    """

    def __init__(self, font_size: int = 15) -> None:
        # Player 0 → left-aligned buffs
        self._bar_p0 = PowerUpTimerBar(align="left", font_size=font_size)
        # Player 1 → right-aligned buffs
        self._bar_p1 = PowerUpTimerBar(align="right", font_size=font_size)
        # Debuffs on opponent side: player-0 debuffs shown right-of-centre,
        # player-1 debuffs shown left-of-centre
        self._badge_from_p0 = PowerUpOpponentBadge(align="right", font_size=font_size - 1)
        self._badge_from_p1 = PowerUpOpponentBadge(align="left",  font_size=font_size - 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        screen: pygame.Surface,
        powerup_manager: "PowerUpManager",
        score_bar_rect: pygame.Rect,
        now: float,
    ) -> None:
        """
        Render all powerup HUD elements anchored to the score bar.

        Parameters
        ----------
        score_bar_rect : pygame.Rect
            Bounding rect of the central score+timer pill (from hud.py).
            Pills are placed immediately below it, growing outward.
        now : float
            Current timestamp (time.time()) for animations.
        """
        effects_p0: List[ActiveEffect] = powerup_manager.get_active_effects(0)
        effects_p1: List[ActiveEffect] = powerup_manager.get_active_effects(1)

        # Partition: self-buffs vs opponent-debuffs
        p0_self, p0_onto_p1 = self._partition(effects_p0)
        p1_self, p1_onto_p0 = self._partition(effects_p1)

        y = score_bar_rect.bottom + _BAR_GAP

        # Left edge of central bar → player-0 buffs grow LEFT from there
        left_anchor  = score_bar_rect.left - PILL_SPACING
        # Right edge of central bar → player-1 buffs grow RIGHT from there
        right_anchor = score_bar_rect.right + PILL_SPACING

        # ── Player 0 self-buffs (left of score bar, right-aligned) ──────
        if p0_self:
            total_w = self._bar_p0.total_width(len(p0_self))
            x = left_anchor - total_w
            self._bar_p0.draw(screen, p0_self, x, y, now)

        # ── Player 1 self-buffs (right of score bar, left-aligned) ──────
        if p1_self:
            self._bar_p1.draw(screen, p1_self, right_anchor, y, now)

        # ── Debuffs applied TO player 1 (by player 0) ── right side ─────
        #    Shown beneath P1's own buffs strip on the right, with red border
        if p0_onto_p1:
            debuff_y = y + PILL_H + _BAR_GAP
            self._badge_from_p0.draw(screen, p0_onto_p1, right_anchor, debuff_y, now)

        # ── Debuffs applied TO player 0 (by player 1) ── left side ──────
        if p1_onto_p0:
            total_w = self._bar_p0.total_width(len(p1_onto_p0))
            debuff_y = y + PILL_H + _BAR_GAP
            x = left_anchor - total_w
            self._badge_from_p1.draw(screen, p1_onto_p0, x, debuff_y, now)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _partition(
        effects: "List[ActiveEffect]",
    ) -> "tuple[List[ActiveEffect], List[ActiveEffect]]":
        """
        Split *effects* into (self_buffs, opponent_debuffs).

        A debuff is any effect whose target_idx differs from its collector_idx
        (i.e. it was collected by player X but applied to player Y).
        """
        self_buffs: List[ActiveEffect]     = []
        opponent_debuffs: List[ActiveEffect] = []
        for eff in effects:
            if eff.target_idx == eff.collector_idx:
                self_buffs.append(eff)
            else:
                opponent_debuffs.append(eff)
        return self_buffs, opponent_debuffs
