"""
PowerUpHelpPanel — panel modal scrollable con tarjetas de ayuda de powerups.

Plan §8 — Integración con el Botón de Ayuda                     (T-0.9)

Responsabilidades (SRP):
    PowerUpHelpCard   — renderiza UNA tarjeta  (game/components/PowerUpHelpCard.py)
    PowerUpHelpPanel  — gestiona el overlay, scroll y ciclo de vida del panel
    draw_help_overlay — función de conveniencia para uso desde game_engine

Diseño:
    - Composición: PowerUpHelpPanel contiene una lista de PowerUpHelpCard.
    - Sin estado global: cada instancia es independiente.
    - Extensible: basta con registrar nuevos powerups en PowerUpRegistry.
    - La fuente de verdad es PowerUpRegistry; el panel no tiene datos propios.
    - ScrollableContainer controla el scroll del contenido.
    - Separación corte: el panel no sabe nada del motor de juego.

Apertura / cierre:
    - panel.toggle()  → abre si cerrado, cierra si abierto.
    - panel.close()   → cierra con animación fade-out.
    - panel.is_open   → True mientras esté visible (incluso fade-out).

Animación (slide + fade):
    open:   0.25 s — escala 0.92 → 1.0 con fade-in (ease-out)
    close:  0.20 s — fade-out alpha 255 → 0

Controles:
    Teclado:
        H / Escape → cerrar
        ↑ / ↓      → scroll
        Page Up/Down → scroll rápido
    Ratón:
        Rueda       → scroll (delegado a ScrollableContainer)
        Clic fuera del panel → cerrar
        Hover sobre tarjeta → resaltar
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from game.components.FontCache import font_cache
from game.components.PowerUpHelpCard import PowerUpHelpCard
from game.components.ScrollableContainer import ScrollableContainer

if TYPE_CHECKING:
    from shared.powerups.registry import PowerUpRegistry


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_PANEL_W_RATIO:  float = 0.52    # panel width as fraction of screen width
_PANEL_H_RATIO:  float = 0.80    # panel height as fraction of screen height
_PANEL_MIN_W:    int   = 380
_PANEL_MAX_W:    int   = 620

_HEADER_H:       int   = 52      # fixed header (title + hint)
_FOOTER_H:       int   = 36      # fixed footer (close button)
_CARD_GAP:       int   = 10      # vertical gap between cards
_CARD_MARGIN_X:  int   = 16      # horizontal margin inside scroll area
_SCROLL_SPEED:   int   = 30      # px per keyboard scroll step

_ANIM_OPEN_DUR:  float = 0.25
_ANIM_CLOSE_DUR: float = 0.20

# Colors
_BG_COLOR        = (14, 15, 22, 235)
_BORDER_COLOR    = (60, 80, 120, 200)
_HEADER_COLOR    = (22, 24, 36, 255)
_TITLE_TXT_COLOR = (220, 225, 240)
_HINT_TXT_COLOR  = (120, 130, 150)
_CLOSE_IDLE_C    = (50, 55, 70)
_CLOSE_HOVER_C   = (90, 60, 60)
_CLOSE_BORDER_C  = (160, 80, 80)

# Keyboard fast-scroll
_PAGE_SCROLL_MULT: int = 5


# ---------------------------------------------------------------------------
# PowerUpHelpPanel
# ---------------------------------------------------------------------------

class PowerUpHelpPanel:
    """
    Scrollable overlay panel listing all registered powerup help cards.

    Parameters
    ----------
    registry : PowerUpRegistry
        Source of truth for powerup definitions (sorted by phase).
    screen_w, screen_h : int
        Current screen dimensions (used to compute panel geometry).
    """

    def __init__(
        self,
        registry: "PowerUpRegistry",
        screen_w: int,
        screen_h: int,
    ) -> None:
        self._registry = registry
        self._screen_w = screen_w
        self._screen_h = screen_h

        # Build geometry
        panel_w = max(_PANEL_MIN_W, min(_PANEL_MAX_W, int(screen_w * _PANEL_W_RATIO)))
        panel_h = int(screen_h * _PANEL_H_RATIO)
        px = (screen_w - panel_w) // 2
        py = (screen_h - panel_h) // 2
        self._panel_rect = pygame.Rect(px, py, panel_w, panel_h)

        # Card width fits inside scroll area minus margins
        card_w = panel_w - _CARD_MARGIN_X * 2 - 8  # 8px reserved for scroll indicators

        # Build cards from registry (ordered by phase)
        self._cards: list[PowerUpHelpCard] = [
            PowerUpHelpCard(defn, card_width=card_w)
            for defn in registry.get_all_by_phase()
        ]

        # Compute total content height
        total_content_h = _CARD_MARGIN_X
        for card in self._cards:
            total_content_h += card.measure_height() + _CARD_GAP
        total_content_h += _CARD_MARGIN_X

        # Scroll container (viewport = panel minus header/footer)
        scroll_rect = pygame.Rect(
            px,
            py + _HEADER_H,
            panel_w,
            panel_h - _HEADER_H - _FOOTER_H,
        )
        self._scroll = ScrollableContainer(scroll_rect, scroll_speed=_SCROLL_SPEED)
        self._scroll.content_height = total_content_h

        # State
        self._open:       bool  = False
        self._anim_t:     float = 0.0   # animation time accumulator
        self._anim_phase: str   = "none"  # "open" | "close" | "none"
        self._alpha:      int   = 0

        # Hover tracking (card index or -1)
        self._hovered_card: int = -1

        # Cached overlay surface
        self._overlay: pygame.Surface | None = None
        self._close_btn_rect: pygame.Rect | None = None

        # Fonts
        self._f_title = font_cache.get(None, 20)
        self._f_hint  = font_cache.get(None, 13)
        self._f_close = font_cache.get(None, 14)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """True while the panel is visible (including fade-out animation)."""
        return self._open

    def toggle(self) -> None:
        """Open if closed, close if open."""
        if self._open:
            self.close()
        else:
            self.open()

    def open(self) -> None:
        """Begin opening animation."""
        if self._open:
            return
        self._open = True
        self._anim_t = 0.0
        self._anim_phase = "open"
        self._alpha = 0

    def close(self) -> None:
        """Begin closing animation."""
        if not self._open:
            return
        self._anim_t = 0.0
        self._anim_phase = "close"

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle a pygame event.

        Returns True if the event was consumed (caller should not propagate).
        """
        if not self._open:
            return False

        # Keyboard
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_h, pygame.K_ESCAPE):
                self.close()
                return True
            if event.key == pygame.K_UP:
                self._scroll.scroll_offset = max(
                    0, self._scroll.scroll_offset - _SCROLL_SPEED
                )
                return True
            if event.key == pygame.K_DOWN:
                self._scroll.scroll_offset = min(
                    self._scroll.max_scroll,
                    self._scroll.scroll_offset + _SCROLL_SPEED,
                )
                return True
            if event.key == pygame.K_PAGEUP:
                self._scroll.scroll_offset = max(
                    0,
                    self._scroll.scroll_offset - _SCROLL_SPEED * _PAGE_SCROLL_MULT,
                )
                return True
            if event.key == pygame.K_PAGEDOWN:
                self._scroll.scroll_offset = min(
                    self._scroll.max_scroll,
                    self._scroll.scroll_offset + _SCROLL_SPEED * _PAGE_SCROLL_MULT,
                )
                return True

        # Mouse scroll
        if self._scroll.handle_event(event):
            return True

        # Click outside → close
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self._panel_rect.collidepoint(event.pos):
                self.close()
                return True
            # Close button
            if self._close_btn_rect and self._close_btn_rect.collidepoint(event.pos):
                self.close()
                return True

        return False

    def update(self, dt: float) -> None:
        """Advance animation. Call once per frame with delta-time in seconds."""
        if self._anim_phase == "open":
            self._anim_t += dt
            progress = min(1.0, self._anim_t / _ANIM_OPEN_DUR)
            # ease-out cubic
            ease = 1 - (1 - progress) ** 3
            self._alpha = int(ease * 255)
            if progress >= 1.0:
                self._anim_phase = "none"
                self._alpha = 255

        elif self._anim_phase == "close":
            self._anim_t += dt
            progress = min(1.0, self._anim_t / _ANIM_CLOSE_DUR)
            ease = progress ** 2  # ease-in quad
            self._alpha = int((1 - ease) * 255)
            if progress >= 1.0:
                self._open = False
                self._anim_phase = "none"
                self._alpha = 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the panel onto *screen*.

        Must be called after :meth:`update` each frame.
        """
        if not self._open:
            return

        # ── Dim overlay ──────────────────────────────────────────────
        if self._overlay is None or self._overlay.get_size() != (
            self._screen_w, self._screen_h
        ):
            self._overlay = pygame.Surface(
                (self._screen_w, self._screen_h), pygame.SRCALPHA
            )
        overlay_alpha = int(self._alpha * 0.55)
        self._overlay.fill((0, 0, 0, overlay_alpha))
        screen.blit(self._overlay, (0, 0))

        # ── Scale animation ──────────────────────────────────────────
        scale = 0.92 + 0.08 * (self._alpha / 255)
        pr = self._panel_rect
        w_s = int(pr.width * scale)
        h_s = int(pr.height * scale)
        ox  = pr.centerx - w_s // 2
        oy  = pr.centery - h_s // 2

        # ── Panel background ─────────────────────────────────────────
        panel_surf = pygame.Surface((w_s, h_s), pygame.SRCALPHA)
        bg_a = self._alpha
        panel_surf.fill((*_BG_COLOR[:3], min(235, bg_a)))
        pygame.draw.rect(
            panel_surf, (*_BORDER_COLOR[:3], bg_a),
            (0, 0, w_s, h_s), 2, border_radius=12
        )
        screen.blit(panel_surf, (ox, oy))

        if self._alpha < 30:
            return   # skip content during very early/late animation

        # ── Header ───────────────────────────────────────────────────
        header_surf = pygame.Surface((w_s, _HEADER_H), pygame.SRCALPHA)
        header_surf.fill((*_HEADER_COLOR[:3], min(255, self._alpha)))
        screen.blit(header_surf, (ox, oy))

        title_s = self._f_title.render("📖  Guía de Powerups", True, _TITLE_TXT_COLOR)
        title_s.set_alpha(self._alpha)
        screen.blit(title_s, (
            ox + (w_s - title_s.get_width()) // 2,
            oy + (_HEADER_H - title_s.get_height()) // 2 - 6,
        ))

        hint_s = self._f_hint.render(
            "↑↓ / Rueda  —  Scroll     H / ESC  —  Cerrar", True, _HINT_TXT_COLOR
        )
        hint_s.set_alpha(min(200, self._alpha))
        screen.blit(hint_s, (
            ox + (w_s - hint_s.get_width()) // 2,
            oy + _HEADER_H - hint_s.get_height() - 6,
        ))

        pygame.draw.line(
            screen,
            (*_BORDER_COLOR[:3], self._alpha),
            (ox + 12, oy + _HEADER_H),
            (ox + w_s - 12, oy + _HEADER_H),
        )

        # ── Footer (close button) ─────────────────────────────────────
        close_w, close_h = 120, 28
        close_x = ox + (w_s - close_w) // 2
        close_y = oy + h_s - _FOOTER_H + (_FOOTER_H - close_h) // 2
        self._close_btn_rect = pygame.Rect(close_x, close_y, close_w, close_h)
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = self._close_btn_rect.collidepoint(mouse_pos)
        btn_color = _CLOSE_HOVER_C if is_hovered else _CLOSE_IDLE_C
        pygame.draw.rect(screen, btn_color, self._close_btn_rect, border_radius=6)
        pygame.draw.rect(screen, _CLOSE_BORDER_C, self._close_btn_rect, 1, border_radius=6)
        close_txt = self._f_close.render("Cerrar  (H / ESC)", True, (200, 190, 190))
        close_txt.set_alpha(self._alpha)
        screen.blit(close_txt, (
            close_x + (close_w - close_txt.get_width()) // 2,
            close_y + (close_h - close_txt.get_height()) // 2,
        ))

        pygame.draw.line(
            screen,
            (*_BORDER_COLOR[:3], self._alpha),
            (ox + 12, oy + h_s - _FOOTER_H),
            (ox + w_s - 12, oy + h_s - _FOOTER_H),
        )

        # ── Scrollable card area ──────────────────────────────────────
        # Recompute scroll rect to match scaled panel
        scroll_rect = pygame.Rect(
            ox, oy + _HEADER_H, w_s, h_s - _HEADER_H - _FOOTER_H
        )
        self._scroll.rect = scroll_rect

        # Hover detection (world y → content y)
        self._hovered_card = -1
        if scroll_rect.collidepoint(mouse_pos):
            local_y = (
                mouse_pos[1] - scroll_rect.y + self._scroll.scroll_offset
            )
            cy = _CARD_MARGIN_X
            for idx, card in enumerate(self._cards):
                ch = card.measure_height()
                if cy <= local_y <= cy + ch:
                    self._hovered_card = idx
                    break
                cy += ch + _CARD_GAP

        # Draw cards onto content surface via ScrollableContainer
        content_surf = self._scroll.begin(screen)

        content_surf.fill((0, 0, 0, 0))   # transparent background
        cy = _CARD_MARGIN_X
        card_w = w_s - _CARD_MARGIN_X * 2 - 8
        # Resize cards to current (potentially scaled) width
        for idx, card in enumerate(self._cards):
            if card.card_width != card_w:
                card.card_width = card_w
                card._inner_w = card_w - _CARD_MARGIN_X * 2
                card._help_lines = card._wrap_text_cached()
                card._tip_lines  = card._wrap_tip_cached()
                card._cached_height = None

            highlighted = (idx == self._hovered_card)
            card.draw_at(content_surf, _CARD_MARGIN_X, cy, highlighted=highlighted)
            cy += card.measure_height() + _CARD_GAP

        self._scroll.end()

        # Apply global alpha to the scrolled region by blitting with alpha
        # (pygame doesn't support per-surface alpha for SRCALPHA surfaces easily,
        # so we apply alpha to the panel_surf already done above)

    # ------------------------------------------------------------------
    # Resize support
    # ------------------------------------------------------------------

    def resize(self, screen_w: int, screen_h: int) -> None:
        """Rebuild geometry for a new screen size."""
        self.__init__(self._registry, screen_w, screen_h)
