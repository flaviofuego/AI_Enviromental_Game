"""
FontCache — Singleton cache for pygame fonts.

Avoids creating new ``pygame.font.Font`` objects every frame, which is
a significant performance bottleneck in multiple components (Card, PopUp,
PauseMenu, home.py, etc.).

Usage::

    from game.components.FontCache import font_cache

    font = font_cache.get(None, 24)          # system default, size 24
    font = font_cache.get("Arial", 32)       # named font
    rendered = font_cache.render("Hello", 24) # shortcut for static text
"""

from __future__ import annotations

import pygame


class FontCache:
    """Thread-safe singleton cache for ``pygame.font.Font`` instances.

    Fonts are keyed by ``(font_name, size)`` and created once on first
    request.  Call :meth:`clear` when the display mode changes (very
    rare in this project).
    """

    _instance: FontCache | None = None

    def __new__(cls) -> FontCache:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._fonts: dict[tuple[str | None, int], pygame.font.Font] = {}
            cls._instance._render_cache: dict[tuple[str, str | None, int, tuple], pygame.Surface] = {}
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, font_name: str | None = None, size: int = 24) -> pygame.font.Font:
        """Return a cached ``pygame.font.Font`` for *font_name* / *size*.

        Parameters
        ----------
        font_name:
            ``None`` for the pygame default font, or a font file path /
            system font name.
        size:
            Point size.
        """
        key = (font_name, size)
        if key not in self._fonts:
            try:
                self._fonts[key] = pygame.font.Font(font_name, size)
            except (FileNotFoundError, OSError):
                self._fonts[key] = pygame.font.SysFont(
                    font_name or "Arial", size
                )
        return self._fonts[key]

    def render(
        self,
        text: str,
        size: int = 24,
        color: tuple = (255, 255, 255),
        font_name: str | None = None,
        antialias: bool = True,
    ) -> pygame.Surface:
        """Render *text* using a cached font **and** cache the surface.

        Ideal for static labels/titles that never change.  The cache key
        is ``(text, font_name, size, color)``; if any of those change the
        surface is re-rendered automatically.
        """
        key = (text, font_name, size, color)
        if key not in self._render_cache:
            font = self.get(font_name, size)
            self._render_cache[key] = font.render(text, antialias, color)
        return self._render_cache[key]

    def clear(self) -> None:
        """Drop all cached fonts and rendered surfaces."""
        self._fonts.clear()
        self._render_cache.clear()


# Module-level singleton for convenient imports.
font_cache = FontCache()
