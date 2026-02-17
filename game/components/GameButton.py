"""
GameButton — Universal, optimized button component for the game UI.

Design principles:
- **SRP**: ``update()`` handles input/state; ``draw()`` only renders.
- **Performance**: Surfaces are pre-rendered per state and cached.
  They are only rebuilt when a property changes (``_dirty`` flag).
- **Flexibility**: Supports text, solid-color background, image background,
  borders, icons (via ``IconRenderer``), and animated hover.
- **Event-driven input**: Click detection uses ``MOUSEBUTTONDOWN`` events
  (discrete), never ``pygame.mouse.get_pressed()`` (continuous).
- **Audio integration**: Optional hover/click sounds through ``AudioManager``.

Usage::

    btn = GameButton(
        text="Play",
        size=(150, 40),
        position=(100, 200),
        bg_color=(0, 100, 200),
        hover_color=(0, 150, 255),
        border_radius=6,
    )

    # In the game loop:
    for event in pygame.event.get():
        if btn.update(event):
            do_something()

    btn.draw(screen)
"""
from __future__ import annotations

import enum
import math
from typing import Callable

import pygame

from .FontCache import font_cache

# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class ButtonState(enum.Enum):
    NORMAL = "normal"
    HOVER = "hover"
    PRESSED = "pressed"
    DISABLED = "disabled"


# ---------------------------------------------------------------------------
# GameButton
# ---------------------------------------------------------------------------

class GameButton:
    """Universal button component with cached rendering and separated I/O.

    Parameters
    ----------
    text : str
        Label displayed on the button.
    icon : pygame.Surface | None
        Optional icon surface rendered to the left of the text.
    icon_draw_fn : callable | None
        Alternative: an ``IconRenderer.draw_*`` static method.  Called as
        ``icon_draw_fn(surface, icon_rect, icon_color, padding=icon_padding)``.
    icon_padding : int
        Inner padding passed to *icon_draw_fn*.
    bg_image : pygame.Surface | None
        If provided, the button renders this image instead of a solid rect.
        This path preserves backward-compatibility with the old PNG-based
        ``Button`` class used in *home.py* (circular sprites).
    position : tuple[int, int]
        Top-left corner ``(x, y)`` of the button.  Use :pyattr:`center` to
        position by center instead.
    size : tuple[int, int] | None
        Explicit ``(width, height)``.  When *None* the size is computed
        automatically from text + icon + padding.
    padding : tuple[int, int]
        Horizontal and vertical internal padding ``(px, py)``.
    border_radius : int
        Corner radius for ``pygame.draw.rect``.
    bg_color, hover_color, pressed_color, disabled_color : tuple
        Background colours per state (RGB or RGBA).
    text_color : tuple
        Text colour (same for all states).
    border_color : tuple | None
        If given, a border is drawn around the button rect.
    border_width : int
        Width of the border stroke.
    font_name : str | None
        ``None`` = pygame default font.
    font_size : int
        Point size for the label text.
    hover_scale : float
        Scale factor applied on hover (e.g. 1.05 → 5 % larger).
    animation_speed : float
        Transition time in seconds for hover animation (lerp).
    on_click : Callable | None
        Optional callback invoked on click.
    enabled : bool
        When *False* the button renders in its *disabled* state and does
        not respond to input.
    hover_text : str
        Optional tooltip text shown below the button on hover.
    audio_hover : str
        Sound effect name for hover enter.
    audio_click : str
        Sound effect name for click.
    """

    __slots__ = (
        # content
        "_text", "_icon", "_icon_draw_fn", "_icon_padding",
        "_bg_image", "_original_bg_image",
        # geometry
        "_position", "_size", "_padding", "_border_radius",
        # colours
        "_bg_color", "_hover_color", "_pressed_color",
        "_disabled_color", "_text_color",
        "_border_color", "_border_width",
        # typography
        "_font_name", "_font_size",
        # animation
        "_hover_scale", "_anim_speed",
        "_current_scale", "_target_scale",
        # callbacks / state
        "_on_click", "_enabled", "_state",
        "_hover_text",
        # audio
        "_audio_hover", "_audio_click",
        # caches
        "_dirty", "_cached_surfaces",
        "_font", "_rect", "_base_rect",
        "_hover_text_surface",
        "_icon_color",
        # public extras
        "name",
    )

    def __init__(
        self,
        text: str = "",
        icon: pygame.Surface | None = None,
        icon_draw_fn: Callable | None = None,
        icon_padding: int = 4,
        bg_image: pygame.Surface | None = None,
        position: tuple[int, int] = (0, 0),
        size: tuple[int, int] | None = None,
        padding: tuple[int, int] = (20, 10),
        border_radius: int = 8,
        bg_color: tuple = (60, 60, 80),
        hover_color: tuple = (80, 80, 110),
        pressed_color: tuple = (40, 40, 60),
        disabled_color: tuple = (50, 50, 50),
        text_color: tuple = (255, 255, 255),
        border_color: tuple | None = None,
        border_width: int = 2,
        font_name: str | None = None,
        font_size: int = 24,
        hover_scale: float = 1.05,
        animation_speed: float = 0.15,
        on_click: Callable | None = None,
        enabled: bool = True,
        hover_text: str = "",
        audio_hover: str = "button_hover",
        audio_click: str = "button_click",
        icon_color: tuple = (255, 255, 255),
    ):
        # Content
        self._text = text
        self._icon = icon
        self._icon_draw_fn = icon_draw_fn
        self._icon_padding = icon_padding
        self._original_bg_image = bg_image
        self._bg_image: pygame.Surface | None = None

        # Geometry
        self._position = position
        self._padding = padding
        self._border_radius = border_radius

        # Colours
        self._bg_color = bg_color
        self._hover_color = hover_color
        self._pressed_color = pressed_color
        self._disabled_color = disabled_color
        self._text_color = text_color
        self._border_color = border_color
        self._border_width = border_width
        self._icon_color = icon_color

        # Typography
        self._font_name = font_name
        self._font_size = font_size
        self._font: pygame.font.Font = font_cache.get(font_name, font_size)

        # Animation
        self._hover_scale = hover_scale
        self._anim_speed = animation_speed
        self._current_scale = 1.0
        self._target_scale = 1.0

        # Callback / State
        self._on_click = on_click
        self._enabled = enabled
        self._state = ButtonState.NORMAL
        self._hover_text = hover_text
        self._hover_text_surface: pygame.Surface | None = None

        # Audio
        self._audio_hover = audio_hover
        self._audio_click = audio_click

        # Public extras
        self.name = ""

        # Cache
        self._dirty = True
        self._cached_surfaces: dict[ButtonState, pygame.Surface] = {}
        self._rect = pygame.Rect(0, 0, 0, 0)
        self._base_rect = pygame.Rect(0, 0, 0, 0)

        # Determine size
        self._size = size  # may be None
        self._rebuild_cache()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def rect(self) -> pygame.Rect:
        """Current bounding rect (accounts for scale animation)."""
        return self._rect

    @property
    def base_rect(self) -> pygame.Rect:
        """Rect at scale 1.0 — useful for layout calculations."""
        return self._base_rect

    @property
    def state(self) -> ButtonState:
        return self._state

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def text(self) -> str:
        return self._text

    @property
    def center(self) -> tuple[int, int]:
        return self._base_rect.center

    @center.setter
    def center(self, pos: tuple[int, int]):
        """Reposition the button by its center."""
        self._position = (
            pos[0] - self._base_rect.width // 2,
            pos[1] - self._base_rect.height // 2,
        )
        self._base_rect.topleft = self._position
        self._rect = self._base_rect.copy()

    # ------------------------------------------------------------------
    # Public mutators (invalidate cache)
    # ------------------------------------------------------------------

    def set_text(self, text: str) -> None:
        if text != self._text:
            self._text = text
            self._dirty = True

    def set_enabled(self, enabled: bool) -> None:
        if enabled != self._enabled:
            self._enabled = enabled
            self._state = ButtonState.NORMAL if enabled else ButtonState.DISABLED

    def set_position(self, pos: tuple[int, int]) -> None:
        self._position = pos
        self._base_rect.topleft = pos
        self._rect = self._base_rect.copy()

    def set_bg_color(self, color: tuple) -> None:
        if color != self._bg_color:
            self._bg_color = color
            self._dirty = True

    def set_hover_color(self, color: tuple) -> None:
        if color != self._hover_color:
            self._hover_color = color
            self._dirty = True

    # ------------------------------------------------------------------
    # Update (input processing)
    # ------------------------------------------------------------------

    def update(self, event: pygame.event.Event, dt: float = 0.0) -> bool:
        """Process a single pygame event.

        Returns ``True`` when the button is clicked.  Call once per event
        in the event loop (not once per frame).

        Parameters
        ----------
        event : pygame.event.Event
            The event to process.
        dt : float
            Delta time since last frame, used for hover animation.
            Pass ``0`` if you don't need smooth animation.
        """
        if not self._enabled:
            return False

        if self._dirty:
            self._rebuild_cache()

        clicked = False

        if event.type == pygame.MOUSEMOTION:
            was_hover = self._state == ButtonState.HOVER
            if self._base_rect.collidepoint(event.pos):
                if not was_hover:
                    self._state = ButtonState.HOVER
                    self._target_scale = self._hover_scale
                    self._play_sound(self._audio_hover, volume=0.1)
            else:
                if was_hover or self._state == ButtonState.PRESSED:
                    self._state = ButtonState.NORMAL
                    self._target_scale = 1.0

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._base_rect.collidepoint(event.pos):
                self._state = ButtonState.PRESSED
                self._target_scale = 0.97
                self._play_sound(self._audio_click, volume=0.3)
                clicked = True
                if self._on_click:
                    self._on_click()

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._state == ButtonState.PRESSED:
                if self._base_rect.collidepoint(event.pos):
                    self._state = ButtonState.HOVER
                    self._target_scale = self._hover_scale
                else:
                    self._state = ButtonState.NORMAL
                    self._target_scale = 1.0

        # Smooth scale animation
        if dt > 0 and self._current_scale != self._target_scale:
            speed = 1.0 / max(self._anim_speed, 0.01)
            diff = self._target_scale - self._current_scale
            self._current_scale += diff * min(dt * speed, 1.0)
            if abs(self._current_scale - self._target_scale) < 0.001:
                self._current_scale = self._target_scale

        return clicked

    def update_animation(self, dt: float) -> None:
        """Advance hover scale animation without processing events.

        Call once per frame *after* the event loop so the animation
        progresses smoothly between discrete events.
        """
        if dt > 0 and self._current_scale != self._target_scale:
            speed = 1.0 / max(self._anim_speed, 0.01)
            diff = self._target_scale - self._current_scale
            self._current_scale += diff * min(dt * speed, 1.0)
            if abs(self._current_scale - self._target_scale) < 0.001:
                self._current_scale = self._target_scale

    # ------------------------------------------------------------------
    # Draw (rendering only)
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        """Render the button onto *surface* using cached surfaces."""
        if self._dirty:
            self._rebuild_cache()

        state = self._state if self._enabled else ButtonState.DISABLED

        if self._bg_image is not None:
            # Image-based button (backward compat with circular Button)
            self._draw_image_button(surface, state)
        else:
            # Rect-based button (the standard path)
            self._draw_rect_button(surface, state)

    # ------------------------------------------------------------------
    # Convenience: check hover (for external tooltip logic)
    # ------------------------------------------------------------------

    def is_hovered(self) -> bool:
        return self._state in (ButtonState.HOVER, ButtonState.PRESSED)

    # ------------------------------------------------------------------
    # Private: cache rebuild
    # ------------------------------------------------------------------

    def _rebuild_cache(self) -> None:
        """Pre-render a surface for each button state."""
        self._dirty = False

        # Resolve bg_image
        if self._original_bg_image is not None:
            self._bg_image = self._original_bg_image
            if self._bg_image.get_flags() & pygame.SRCALPHA:
                self._bg_image = self._bg_image.convert_alpha()
            else:
                self._bg_image = self._bg_image.convert()

        # Determine size
        if self._size is not None:
            w, h = self._size
        elif self._bg_image is not None:
            w, h = self._bg_image.get_size()
        else:
            w, h = self._compute_auto_size()

        self._base_rect = pygame.Rect(self._position[0], self._position[1], w, h)
        self._rect = self._base_rect.copy()

        # Pre-render hover text
        if self._hover_text:
            self._hover_text_surface = font_cache.render(
                self._hover_text, 14, (255, 255, 255))
        else:
            self._hover_text_surface = None

        # Build per-state surfaces (only for rect-based buttons)
        if self._bg_image is None:
            self._cached_surfaces.clear()
            color_map = {
                ButtonState.NORMAL: self._bg_color,
                ButtonState.HOVER: self._hover_color,
                ButtonState.PRESSED: self._pressed_color,
                ButtonState.DISABLED: self._disabled_color,
            }
            for st, bg in color_map.items():
                self._cached_surfaces[st] = self._render_state_surface(
                    w, h, bg)

    def _compute_auto_size(self) -> tuple[int, int]:
        """Compute button size from text + icon + padding."""
        px, py = self._padding
        tw, th = 0, 0
        if self._text:
            tw, th = self._font.size(self._text)
        icon_w = 0
        if self._icon is not None:
            icon_w = self._icon.get_width() + 8
        elif self._icon_draw_fn is not None:
            icon_w = th + 8 if th else 24  # square icon
        w = tw + icon_w + px * 2
        h = max(th, 20) + py * 2
        return w, h

    def _render_state_surface(
        self, w: int, h: int, bg_color: tuple
    ) -> pygame.Surface:
        """Render a single state onto a new surface."""
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf = surf.convert_alpha()

        # Background
        pygame.draw.rect(surf, bg_color, (0, 0, w, h),
                         border_radius=self._border_radius)

        # Border
        if self._border_color is not None and self._border_width > 0:
            pygame.draw.rect(
                surf, self._border_color, (0, 0, w, h),
                self._border_width, border_radius=self._border_radius)

        # Content (icon + text) centered
        self._blit_content(surf, w, h)
        return surf

    def _blit_content(self, surf: pygame.Surface, w: int, h: int) -> None:
        """Draw icon + text centered on *surf*."""
        # Measure content width
        text_surf = None
        tw, th = 0, 0
        if self._text:
            text_surf = self._font.render(self._text, True, self._text_color)
            tw, th = text_surf.get_size()

        icon_w, icon_h = 0, 0
        gap = 8 if (self._text and (self._icon or self._icon_draw_fn)) else 0
        if self._icon is not None:
            icon_w, icon_h = self._icon.get_size()
        elif self._icon_draw_fn is not None:
            icon_h = th if th else 20
            icon_w = icon_h

        total_w = icon_w + gap + tw
        start_x = (w - total_w) // 2
        cy = h // 2

        # Icon
        if self._icon is not None:
            surf.blit(self._icon, (start_x, cy - icon_h // 2))
        elif self._icon_draw_fn is not None:
            icon_rect = pygame.Rect(start_x, cy - icon_h // 2, icon_w, icon_h)
            self._icon_draw_fn(
                surf, icon_rect, self._icon_color,
                padding=self._icon_padding)

        # Text
        if text_surf is not None:
            surf.blit(text_surf, (start_x + icon_w + gap, cy - th // 2))

    # ------------------------------------------------------------------
    # Private: draw helpers
    # ------------------------------------------------------------------

    def _draw_rect_button(
        self, surface: pygame.Surface, state: ButtonState
    ) -> None:
        """Blit the pre-rendered surface with optional scale animation."""
        cached = self._cached_surfaces.get(state)
        if cached is None:
            return

        if abs(self._current_scale - 1.0) < 0.002:
            # No scale — fast path
            self._rect = self._base_rect.copy()
            surface.blit(cached, self._base_rect)
        else:
            # Scaled (hover / press animation)
            sw = int(self._base_rect.width * self._current_scale)
            sh = int(self._base_rect.height * self._current_scale)
            scaled = pygame.transform.smoothscale(cached, (sw, sh))
            self._rect = scaled.get_rect(center=self._base_rect.center)
            surface.blit(scaled, self._rect)

        # Hover tooltip
        if state in (ButtonState.HOVER, ButtonState.PRESSED):
            self._draw_hover_text(surface)

    def _draw_image_button(
        self, surface: pygame.Surface, state: ButtonState
    ) -> None:
        """Draw an image-based button (circular PNG sprites)."""
        img = self._bg_image
        if img is None:
            return

        if state == ButtonState.HOVER or state == ButtonState.PRESSED:
            scale = self._current_scale
        else:
            scale = 1.0

        if abs(scale - 1.0) > 0.002:
            sw = int(self._base_rect.width * scale)
            sh = int(self._base_rect.height * scale)
            img = pygame.transform.smoothscale(self._bg_image, (sw, sh))
            self._rect = img.get_rect(center=self._base_rect.center)
        else:
            self._rect = self._base_rect.copy()

        surface.blit(img, self._rect)

        # Hover tooltip
        if state in (ButtonState.HOVER, ButtonState.PRESSED):
            self._draw_hover_text(surface)

    def _draw_hover_text(self, surface: pygame.Surface) -> None:
        if self._hover_text_surface is not None:
            hr = self._hover_text_surface.get_rect(
                centerx=self._base_rect.centerx,
                top=self._base_rect.bottom + 4)
            surface.blit(self._hover_text_surface, hr)

    # ------------------------------------------------------------------
    # Private: audio
    # ------------------------------------------------------------------

    @staticmethod
    def _play_sound(name: str, volume: float = 0.3) -> None:
        try:
            from .AudioManager import audio_manager

            audio_manager.play_sound_effect(name, volume_override=volume)
        except Exception:
            pass  # audio is best-effort


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def image_button(
    asset_name: str,
    scale: tuple[int, int],
    position: tuple[int, int],
    hover_text: str = "",
    name: str = "",
) -> GameButton:
    """Create a ``GameButton`` that mimics the legacy ``Button`` (PNG sprite).

    Use this to replace ``Button(image, scale, position, tex_hover)``
    one-for-one during migration.

    Parameters
    ----------
    asset_name : str
        Relative name inside ``game/assets/`` **without** ``.png``.
    scale : tuple
        Desired ``(width, height)`` for the image.
    position : tuple
        Center position ``(cx, cy)``.
    hover_text : str
        Tooltip text shown on hover.
    name : str
        Identifier for external lookup (e.g. ``"play"``).
    """
    import os

    path = f"game/assets/{asset_name}.png"
    if os.path.exists(path):
        raw = pygame.image.load(path)
    else:
        raw = pygame.image.load("game/assets/play.png")  # fallback

    img = pygame.transform.smoothscale(raw, scale)

    btn = GameButton(
        bg_image=img,
        size=scale,
        position=(position[0] - scale[0] // 2, position[1] - scale[1] // 2),
        hover_text=hover_text,
        hover_scale=1.10,
    )
    # Expose a ``.name`` attribute for dict-key mapping (like old Button)
    btn.name = name or asset_name  # type: ignore[attr-defined]
    return btn


def text_button(
    text: str,
    position: tuple[int, int],
    size: tuple[int, int] | None = None,
    *,
    font_size: int = 20,
    bg_color: tuple = (0, 100, 200),
    hover_color: tuple = (0, 150, 255),
    pressed_color: tuple | None = None,
    text_color: tuple = (255, 255, 255),
    border_color: tuple | None = (255, 255, 255),
    border_width: int = 2,
    border_radius: int = 6,
    padding: tuple[int, int] = (16, 8),
    icon_draw_fn: Callable | None = None,
    icon_color: tuple = (255, 255, 255),
    icon_padding: int = 4,
    enabled: bool = True,
    hover_scale: float = 1.0,
    on_click: Callable | None = None,
) -> GameButton:
    """Convenience factory for a rect+text button (the most common case).

    Parameters match the manual pattern used in *pvp_setup.py*,
    *Level_Select.py*, *profile_manager.py*, etc.
    """
    return GameButton(
        text=text,
        position=position,
        size=size,
        font_size=font_size,
        padding=padding,
        bg_color=bg_color,
        hover_color=hover_color,
        pressed_color=pressed_color or _darken(bg_color, 0.7),
        text_color=text_color,
        border_color=border_color,
        border_width=border_width,
        border_radius=border_radius,
        icon_draw_fn=icon_draw_fn,
        icon_color=icon_color,
        icon_padding=icon_padding,
        enabled=enabled,
        hover_scale=hover_scale,
        on_click=on_click,
    )


def _darken(color: tuple, factor: float) -> tuple:
    """Return a darkened copy of *color*."""
    return tuple(max(0, int(c * factor)) for c in color[:3])
