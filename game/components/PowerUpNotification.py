"""
PowerUpNotification — Toast animado de activación/expiración de powerup.

Plan §5.1 — Notificaciones Visuales                            (T-0.8)

Responsabilidades (SRP):
    PowerUpNotification   — estado y ciclo de vida de UN toast
    NotificationQueue     — cola de toasts activos por jugador
    draw_notifications    — función de conveniencia: renderiza toda la cola

Diseño:
    - Composición sobre herencia: NotificationQueue contiene listas de
      PowerUpNotification, no hereda de ninguna clase base.
    - Sin estado global: cada instancia es independiente.
    - Extensible: basta con llamar .push() con una PowerUpDefinition.
    - Frame-independent: toda animación usa `dt` en segundos.

Anatomía del toast
──────────────────
    ┌─────────────────────────────────────┐
    │  ⚡  Viento Solar   │  +30% golpe  │
    └─────────────────────────────────────┘
    · Fondo del color del powerup con alpha configurable (200 por defecto).
    · Franja izquierda opaca del mismo color (4 px) como acento visual.
    · Ícono grande a la izquierda.
    · Nombre y descripción en dos líneas compactas.
    · Badge "COLECTADO" / "EXPIRADO" en esquina superior-derecha.

Animación (Slow-In / Slow-Out — principio Disney §6)
──────────────────────────────────────────────────────
    slide-in:   0.25 s — ease-out (rápido al principio, amortigua al final)
    hold:       2.0  s — visible estático
    fade-out:   0.25 s — ease-in alpha 255→0
    Total:      2.5  s  (DISPLAY_DURATION)

Posición en pantalla
─────────────────────
    "left"  → jugador 1 — aparece en la zona top-left del campo
    "right" → jugador 2 — aparece en la zona top-right del campo
    Los toasts se apilan verticalmente con un pequeño gap.
"""
from __future__ import annotations

import math
from typing import List, Optional, TYPE_CHECKING

import pygame

from game.components.FontCache import font_cache
from game.components.IconRenderer import IconRenderer

if TYPE_CHECKING:
    from shared.powerups.registry import PowerUpDefinition


# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

SLIDE_IN_DURATION:  float = 0.25   # s — ease-out slide from top
HOLD_DURATION:      float = 2.00   # s — fully visible
FADE_OUT_DURATION:  float = 0.25   # s — alpha fade to 0
DISPLAY_DURATION:   float = SLIDE_IN_DURATION + HOLD_DURATION + FADE_OUT_DURATION  # 2.5 s

# Layout
TOAST_WIDTH:   int = 230   # px
TOAST_HEIGHT:  int = 52    # px
TOAST_MARGIN:  int = 8     # gap between stacked toasts
TOAST_PADDING: int = 8     # inner content padding

# Visual
_ACCENT_W:     int = 4     # left-accent stripe width
_ICON_FONT_SZ: int = 22    # large icon on the left
_NAME_FONT_SZ: int = 14    # powerup name line
_DESC_FONT_SZ: int = 12    # description line (smaller)
_BADGE_FONT_SZ: int = 10   # COLLECTED / EXPIRED badge
_BG_ALPHA_MAX: int = 210   # maximum toast background alpha

# Badge colours
_BADGE_COLLECTED  = (80, 220, 120)    # green
_BADGE_EXPIRED    = (180, 80, 80)     # red-muted


# ---------------------------------------------------------------------------
# PowerUpNotification
# ---------------------------------------------------------------------------

class PowerUpNotification:
    """
    A single toast notification for one powerup event.

    States (internal):
        slide_in  → 0..SLIDE_IN_DURATION
        hold      → SLIDE_IN_DURATION..SLIDE_IN_DURATION+HOLD_DURATION
        fade_out  → remainder

    Parameters
    ----------
    definition : PowerUpDefinition
        Source of color, icon, name, description.
    side : "left" | "right"
        Which player's side this toast belongs to.
    event_type : "collected" | "expired"
        Drives the badge label.
    screen_w : int
        Full display width, used to anchor `right` toasts.
    field_top : int
        Y-pixel where the playfield starts, used to anchor toasts below HUD.
    """

    def __init__(
        self,
        definition:  "PowerUpDefinition",
        side:        str,
        event_type:  str,
        screen_w:    int,
        field_top:   int = 0,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if event_type not in ("collected", "expired"):
            raise ValueError("event_type must be 'collected' or 'expired'")

        self.definition  = definition
        self.side        = side
        self.event_type  = event_type
        self._screen_w   = screen_w
        self._field_top  = field_top

        self._elapsed    = 0.0
        self._done       = False

        # Cached surfaces (built lazily on first draw)
        self._bg_surf:    Optional[pygame.Surface] = None
        self._name_surf:  Optional[pygame.Surface] = None
        self._desc_surf:  Optional[pygame.Surface] = None
        self._icon_surf:  Optional[pygame.Surface] = None
        self._badge_surf: Optional[pygame.Surface] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_done(self) -> bool:
        """True once the full animation cycle has completed."""
        return self._done

    def update(self, dt: float) -> None:
        """Advance the animation by *dt* seconds."""
        if self._done:
            return
        self._elapsed += dt
        if self._elapsed >= DISPLAY_DURATION:
            self._done = True

    def draw(self, screen: pygame.Surface, slot: int) -> None:
        """
        Render this toast onto *screen*.

        Parameters
        ----------
        slot : int
            Vertical slot index (0 = top-most). Used to stack multiple toasts.
        """
        if self._done:
            return

        alpha, offset_y = self._compute_animation()
        if alpha <= 0:
            return

        # Determine pixel position
        x, y = self._compute_position(slot, offset_y)

        # Draw body
        self._draw_body(screen, x, y, alpha)
        # Draw accent stripe
        self._draw_accent(screen, x, y, alpha)
        # Draw icon
        self._draw_icon(screen, x, y, alpha)
        # Draw text lines
        self._draw_text(screen, x, y, alpha)
        # Draw badge
        self._draw_badge(screen, x, y, alpha)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _compute_animation(self) -> tuple[int, int]:
        """
        Return (alpha: int, y_offset: int).

        y_offset is used for the slide-in / slide-out effect, measured in
        pixels. Positive = shifted further down (starts above and slides in).
        """
        t = self._elapsed

        if t < SLIDE_IN_DURATION:
            # Ease-out: toast slides in from above
            progress = t / SLIDE_IN_DURATION            # 0 → 1
            eased    = 1.0 - (1.0 - progress) ** 2     # quadratic ease-out
            offset_y = int((1.0 - eased) * -TOAST_HEIGHT)
            alpha    = int(eased * _BG_ALPHA_MAX)

        elif t < SLIDE_IN_DURATION + HOLD_DURATION:
            # Fully visible hold phase
            offset_y = 0
            alpha    = _BG_ALPHA_MAX

        else:
            # Ease-in fade out
            fade_t   = t - SLIDE_IN_DURATION - HOLD_DURATION
            progress = fade_t / FADE_OUT_DURATION       # 0 → 1
            eased    = progress ** 2                    # quadratic ease-in
            offset_y = 0
            alpha    = int((1.0 - eased) * _BG_ALPHA_MAX)

        return max(0, min(255, alpha)), offset_y

    # ------------------------------------------------------------------
    # Positioning
    # ------------------------------------------------------------------

    def _compute_position(self, slot: int, offset_y: int) -> tuple[int, int]:
        """Return top-left (x, y) for this toast given *slot* and *offset_y*."""
        margin = 12   # gap from screen edge
        y_base = self._field_top + margin
        y      = y_base + slot * (TOAST_HEIGHT + TOAST_MARGIN) + offset_y

        if self.side == "left":
            x = margin
        else:
            x = self._screen_w - TOAST_WIDTH - margin

        return x, y

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_body(self, screen: pygame.Surface, x: int, y: int, alpha: int) -> None:
        """Semi-transparent rounded background panel."""
        r, g, b = self.definition.color
        # Darken background for readability while keeping the hue
        bg_r = max(0, r - 80)
        bg_g = max(0, g - 80)
        bg_b = max(0, b - 80)

        surf = pygame.Surface((TOAST_WIDTH, TOAST_HEIGHT), pygame.SRCALPHA)
        surf.fill((bg_r, bg_g, bg_b, alpha))
        screen.blit(surf, (x, y))

        # Thin border in the powerup color
        border_alpha = min(255, int(alpha * 1.2))
        border_color = (r, g, b, border_alpha)
        pygame.draw.rect(
            screen, border_color,
            (x, y, TOAST_WIDTH, TOAST_HEIGHT),
            1, border_radius=4
        )

    def _draw_accent(self, screen: pygame.Surface, x: int, y: int, alpha: int) -> None:
        """Solid left-side color accent stripe."""
        r, g, b = self.definition.color
        accent_surf = pygame.Surface((_ACCENT_W, TOAST_HEIGHT), pygame.SRCALPHA)
        accent_surf.fill((r, g, b, min(255, int(alpha * 1.3))))
        screen.blit(accent_surf, (x, y))

    def _draw_icon(self, screen: pygame.Surface, x: int, y: int, alpha: int) -> None:
        """Large powerup icon to the left of the text."""
        if self._icon_surf is None:
            self._icon_surf = IconRenderer.get_powerup_surface(
                self.definition.icon, _ICON_FONT_SZ, (255, 255, 255)
            )

        # Apply alpha tinting by blending with a per-pixel-alpha copy
        icon = self._icon_surf.copy()
        icon.set_alpha(alpha)
        ix = x + _ACCENT_W + TOAST_PADDING
        iy = y + (TOAST_HEIGHT - icon.get_height()) // 2
        screen.blit(icon, (ix, iy))

    def _draw_text(self, screen: pygame.Surface, x: int, y: int, alpha: int) -> None:
        """Name and description lines."""
        if self._name_surf is None:
            font_name = font_cache.get(None, _NAME_FONT_SZ)
            font_desc = font_cache.get(None, _DESC_FONT_SZ)
            self._name_surf = font_name.render(
                self.definition.name, True, (255, 255, 255)
            )
            self._desc_surf = font_desc.render(
                self.definition.description, True, (210, 210, 210)
            )

        icon_right = x + _ACCENT_W + TOAST_PADDING + _ICON_FONT_SZ + TOAST_PADDING
        text_area_w = TOAST_WIDTH - (icon_right - x) - TOAST_PADDING

        # Name line (vertically centered in upper half)
        name = self._name_surf.copy()
        name.set_alpha(alpha)
        ny = y + TOAST_PADDING
        # Clip to text area width
        clip_w = min(text_area_w, name.get_width())
        screen.blit(name, (icon_right, ny), area=(0, 0, clip_w, name.get_height()))

        # Description line
        desc = self._desc_surf.copy()
        desc.set_alpha(alpha)
        dy = ny + name.get_height() + 2
        clip_w = min(text_area_w, desc.get_width())
        screen.blit(desc, (icon_right, dy), area=(0, 0, clip_w, desc.get_height()))

    def _draw_badge(self, screen: pygame.Surface, x: int, y: int, alpha: int) -> None:
        """Small badge in top-right corner: COLECTADO / EXPIRADO."""
        if self._badge_surf is None:
            label = "COLECTADO" if self.event_type == "collected" else "EXPIRADO"
            color = _BADGE_COLLECTED if self.event_type == "collected" else _BADGE_EXPIRED
            font  = font_cache.get(None, _BADGE_FONT_SZ)
            self._badge_surf = font.render(label, True, color)

        badge = self._badge_surf.copy()
        badge.set_alpha(int(alpha * 0.85))
        bx = x + TOAST_WIDTH - badge.get_width() - TOAST_PADDING
        by = y + 3
        screen.blit(badge, (bx, by))

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PowerUpNotification("
            f"id={self.definition.id!r}, "
            f"side={self.side!r}, "
            f"event={self.event_type!r}, "
            f"elapsed={self._elapsed:.2f}s)"
        )


# ---------------------------------------------------------------------------
# NotificationQueue
# ---------------------------------------------------------------------------

class NotificationQueue:
    """
    Manages a list of active PowerUpNotification toasts for ONE player side.

    Responsibilities:
        - Accept new toasts via .push()
        - Advance all timers via .update(dt)
        - Evict completed toasts automatically
        - Expose .draw() for rendering all active toasts in stacked order

    Design (SRP):
        This object only manages *which toasts are alive* and *their order*.
        Rendering logic lives in PowerUpNotification.draw().

    Parameters
    ----------
    side : "left" | "right"
        Forwarded to every created PowerUpNotification.
    screen_w : int
        Full display width.
    field_top : int
        Y-pixel where the playing field starts (below HUD bar).
    max_visible : int
        Maximum number of toasts visible simultaneously.  Older toasts
        are silently dropped when the queue is full.
    """

    def __init__(
        self,
        side:         str,
        screen_w:     int,
        field_top:    int = 0,
        max_visible:  int = 3,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self._side        = side
        self._screen_w    = screen_w
        self._field_top   = field_top
        self._max_visible = max_visible
        self._toasts:     List[PowerUpNotification] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        definition: "PowerUpDefinition",
        event_type: str = "collected",
    ) -> None:
        """
        Enqueue a new toast notification.

        If the queue is already at *max_visible* capacity, the oldest
        existing toast is immediately retired to make room.
        """
        if len(self._toasts) >= self._max_visible:
            # Drop the oldest so the newest is always shown
            self._toasts.pop(0)

        toast = PowerUpNotification(
            definition  = definition,
            side        = self._side,
            event_type  = event_type,
            screen_w    = self._screen_w,
            field_top   = self._field_top,
        )
        self._toasts.append(toast)

    def update(self, dt: float) -> None:
        """Advance all toasts and remove completed ones."""
        for toast in self._toasts:
            toast.update(dt)
        self._toasts = [t for t in self._toasts if not t.is_done]

    def draw(self, screen: pygame.Surface) -> None:
        """Render all active toasts in their stacked slots."""
        for slot, toast in enumerate(self._toasts):
            toast.draw(screen, slot)

    @property
    def active_count(self) -> int:
        """Number of toasts currently in the queue."""
        return len(self._toasts)

    def clear(self) -> None:
        """Remove all pending toasts (e.g. on match reset)."""
        self._toasts.clear()

    def __repr__(self) -> str:
        return f"NotificationQueue(side={self._side!r}, active={len(self._toasts)})"


# ---------------------------------------------------------------------------
# Convenience function — one call renders both queues
# ---------------------------------------------------------------------------

def draw_notifications(
    screen:      pygame.Surface,
    queue_left:  NotificationQueue,
    queue_right: NotificationQueue,
) -> None:
    """
    Render all active notifications for both players.

    Intended to be called once per frame after all game elements are drawn,
    so toasts appear on top.

    Example::

        draw_notifications(screen, p1_notifications, p2_notifications)
    """
    queue_left.draw(screen)
    queue_right.draw(screen)
