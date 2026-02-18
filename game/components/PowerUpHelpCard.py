"""
PowerUpHelpCard — tarjeta visual de información de un powerup.

Responsabilidades (SRP):
    PowerUpHelpCard  — renderiza UNA tarjeta de ayuda para UNA definición.

Anatomía
────────
    ┌──────────────────────────────────────────────┐
    │  ⚡  Viento Solar               [Fase 1] 5s  │  ← cabecera
    │  ──────────────────────────────────────────  │
    │  "+30% fuerza de golpe"                       │  ← descripción corta
    │                                               │
    │  El viento solar carga tu mazo de energía     │  ← help_text
    │  cinética, incrementando la potencia de       │
    │  cada golpe. Cronometra su uso...             │
    │                                               │
    │  💡 Recoger justo antes de un ataque para     │  ← strategy_tip
    │     maximizar la fuerza del golpe.            │
    └──────────────────────────────────────────────┘

Diseño:
    - Sin estado mutable: todo se recalcula en draw().
    - El caller controla x, y (scroll externo en PowerUpHelpPanel).
    - Card width se pasa en draw_at(); height expuesto via .measure_height().
    - Frame-independent: sin animación propia (la gestiona el Panel).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from game.components.FontCache import font_cache

if TYPE_CHECKING:
    from shared.powerups.registry import PowerUpDefinition


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

_CARD_PAD_X: int = 14       # horizontal inner padding
_CARD_PAD_Y: int = 12       # vertical inner padding
_HEADER_H:   int = 32       # fixed header row height
_DIVIDER_H:  int = 1        # separator line height
_LINE_GAP:   int = 4        # gap between text lines
_TIP_ICON:   str = "💡"
_CORNER_R:   int = 8        # border radius

# Font sizes
_FS_ICON:   int = 20
_FS_TITLE:  int = 16
_FS_PHASE:  int = 12
_FS_DESC:   int = 13
_FS_BODY:   int = 12
_FS_TIP:    int = 12


def _wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    """Break *text* into lines that fit within *max_width* pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if font.size(test)[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


class PowerUpHelpCard:
    """
    Stateless renderer for one :class:`~shared.powerups.registry.PowerUpDefinition`.

    Parameters
    ----------
    definition : PowerUpDefinition
        Source of all display data.
    card_width : int
        Pixel width of the rendered card (panel controls this).
    """

    def __init__(self, definition: "PowerUpDefinition", card_width: int = 340) -> None:
        self.definition = definition
        self.card_width = card_width

        # Pre-load fonts (FontCache → zero allocation after first call)
        self._f_icon  = font_cache.get(None, _FS_ICON)
        self._f_title = font_cache.get(None, _FS_TITLE)
        self._f_phase = font_cache.get(None, _FS_PHASE)
        self._f_desc  = font_cache.get(None, _FS_DESC)
        self._f_body  = font_cache.get(None, _FS_BODY)
        self._f_tip   = font_cache.get(None, _FS_TIP)

        # Precompute wrapped lines (card_width - 2*pad)
        self._inner_w: int = card_width - _CARD_PAD_X * 2
        self._help_lines: list[str] = _wrap_text(
            definition.help_text, self._f_body, self._inner_w
        )
        self._tip_lines: list[str] = _wrap_text(
            f"  {definition.strategy_tip}", self._f_tip, self._inner_w - 20
        )
        # Cache computed height
        self._cached_height: int | None = None

    # ------------------------------------------------------------------
    # Cache-refresh helpers (called by PowerUpHelpPanel on resize)
    # ------------------------------------------------------------------

    def _wrap_text_cached(self) -> list[str]:
        """Re-wrap help_text for the current card_width."""
        inner = self.card_width - _CARD_PAD_X * 2
        return _wrap_text(self.definition.help_text, self._f_body, inner)

    def _wrap_tip_cached(self) -> list[str]:
        """Re-wrap strategy_tip for the current card_width."""
        inner = self.card_width - _CARD_PAD_X * 2
        return _wrap_text(f"  {self.definition.strategy_tip}", self._f_tip, inner - 20)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_height(self) -> int:
        """Return the total pixel height this card will occupy."""
        if self._cached_height is not None:
            return self._cached_height

        h = _CARD_PAD_Y
        h += _HEADER_H                          # icon + name row
        h += _DIVIDER_H + _LINE_GAP             # separator
        h += self._f_desc.get_linesize()        # description
        h += _LINE_GAP * 2
        h += len(self._help_lines) * (self._f_body.get_linesize() + _LINE_GAP)
        if self._tip_lines:
            h += _LINE_GAP
            h += len(self._tip_lines) * (self._f_tip.get_linesize() + _LINE_GAP)
        h += _CARD_PAD_Y

        self._cached_height = h
        return h

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw_at(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        *,
        highlighted: bool = False,
    ) -> None:
        """
        Draw the card onto *surface* with its top-left at *(x, y)*.

        Parameters
        ----------
        highlighted : bool
            Draw an accent border when True (e.g. mouse-over).
        """
        defn = self.definition
        w    = self.card_width
        h    = self.measure_height()
        r, g, b = defn.color

        # --- Background ---
        card_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        bg_alpha  = 200 if highlighted else 160
        card_surf.fill((20, 22, 30, bg_alpha))

        # Accent left stripe in powerup color
        stripe = pygame.Rect(0, 0, 4, h)
        pygame.draw.rect(card_surf, (r, g, b, 220), stripe)

        # Border
        border_color = (r, g, b, 220 if highlighted else 120)
        pygame.draw.rect(card_surf, border_color, (0, 0, w, h), 1, border_radius=_CORNER_R)

        surface.blit(card_surf, (x, y))

        # --- Header row ---
        cy = y + _CARD_PAD_Y

        # Icon
        icon_surf = self._f_icon.render(defn.icon, True, (r, g, b))
        surface.blit(icon_surf, (x + _CARD_PAD_X, cy + (_HEADER_H - icon_surf.get_height()) // 2))

        # Name
        name_surf = self._f_title.render(defn.name, True, (230, 230, 240))
        nx = x + _CARD_PAD_X + icon_surf.get_width() + 8
        surface.blit(name_surf, (nx, cy + (_HEADER_H - name_surf.get_height()) // 2))

        # Phase + duration badge (right-aligned)
        badge_str = f"F{defn.phase}  {defn.duration:.0f}s"
        badge_surf = self._f_phase.render(badge_str, True, (r, g, b))
        bx = x + w - _CARD_PAD_X - badge_surf.get_width()
        surface.blit(badge_surf, (bx, cy + (_HEADER_H - badge_surf.get_height()) // 2))

        cy += _HEADER_H

        # --- Divider ---
        pygame.draw.line(
            surface, (r, g, b, 80),
            (x + _CARD_PAD_X, cy), (x + w - _CARD_PAD_X, cy),
        )
        cy += _DIVIDER_H + _LINE_GAP

        # --- Short description ---
        desc_surf = self._f_desc.render(defn.description, True, (r, g, b))
        surface.blit(desc_surf, (x + _CARD_PAD_X, cy))
        cy += self._f_desc.get_linesize() + _LINE_GAP * 2

        # --- Help text (wrapped) ---
        for line in self._help_lines:
            ls = self._f_body.render(line, True, (190, 195, 210))
            surface.blit(ls, (x + _CARD_PAD_X, cy))
            cy += self._f_body.get_linesize() + _LINE_GAP

        # --- Strategy tip ---
        if self._tip_lines:
            cy += _LINE_GAP
            tip_icon_surf = self._f_tip.render(_TIP_ICON, True, (255, 220, 80))
            surface.blit(tip_icon_surf, (x + _CARD_PAD_X, cy))
            for i, line in enumerate(self._tip_lines):
                tip_s = self._f_tip.render(line, True, (210, 200, 160))
                surface.blit(tip_s, (x + _CARD_PAD_X + 20, cy))
                cy += self._f_tip.get_linesize() + _LINE_GAP
