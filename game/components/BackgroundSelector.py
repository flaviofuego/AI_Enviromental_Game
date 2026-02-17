"""
Level-background selector component for PvP setup.
Displays a horizontal row of level thumbnails; clicking one selects
the background for the match.
"""
import pygame
from .LevelThumbnail import LevelThumbnail
from .AudioManager import audio_manager
from ..config.level_config import LEVELS


# Pre-built short names for each level
_LEVEL_NAMES: dict[int, str] = {
    1: "Artico",
    2: "Ozono",
    3: "Smog",
    4: "Bosque",
    5: "Calor Urbano",
}


class BackgroundSelector:
    """Horizontal row of clickable level thumbnails.

    Parameters
    ----------
    level_ids : list[int] | None
        IDs to show. Defaults to 1-5.
    thumb_size : tuple[int, int]
        Size of each thumbnail.
    spacing : int
        Horizontal gap between thumbnails.
    selected_id : int
        Initially selected level.
    label_font : pygame.font.Font | None
        Font for the name below each thumbnail.
    """

    def __init__(
        self,
        level_ids: list[int] | None = None,
        thumb_size: tuple[int, int] = (120, 70),
        spacing: int = 10,
        selected_id: int = 1,
        label_font: pygame.font.Font | None = None,
    ):
        self.level_ids = level_ids or list(range(1, 6))
        self.thumb_size = thumb_size
        self.spacing = spacing
        self.selected_id = selected_id
        self.label_font = label_font or pygame.font.Font(None, 16)

        # Pre-load thumbnails
        self._thumbnails: dict[int, LevelThumbnail] = {}
        for lid in self.level_ids:
            self._thumbnails[lid] = LevelThumbnail(lid, size=thumb_size)

        # Rects populated during draw for hit-testing
        self._rects: list[dict] = []

        # Selection ring colors
        self._select_color = (255, 215, 0)  # gold
        self._hover_color = (173, 216, 230)  # ice-blue

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface, x: int, y: int,
             max_width: int | None = None) -> int:
        """Draw the selector. Returns total height consumed.

        If *max_width* is given and thumbnails would overflow, they are
        wrapped into multiple rows.
        """
        self._rects.clear()
        tw, th = self.thumb_size
        label_h = self.label_font.get_linesize() + 4
        cell_w = tw + self.spacing
        cell_h = th + label_h + 8 + self.spacing

        cols_per_row = len(self.level_ids)
        if max_width:
            cols_per_row = max(1, (max_width + self.spacing) // cell_w)

        row = 0
        col = 0
        for lid in self.level_ids:
            tx = x + col * cell_w
            ty = y + row * cell_h

            thumb = self._thumbnails[lid]

            # Selection highlight
            is_selected = lid == self.selected_id
            if is_selected:
                highlight = pygame.Rect(tx - 3, ty - 3, tw + 6, th + 6)
                pygame.draw.rect(surface, self._select_color, highlight, 3, border_radius=4)

            thumb.draw(surface, (tx, ty))

            # Level name label
            name = _LEVEL_NAMES.get(lid, f"Nivel {lid}")
            lbl = self.label_font.render(name, True, (255, 255, 255))
            surface.blit(lbl, lbl.get_rect(centerx=tx + tw // 2, top=ty + th + 4))

            self._rects.append({
                "rect": pygame.Rect(tx, ty, tw, th + label_h + 4),
                "level_id": lid,
            })

            col += 1
            if col >= cols_per_row:
                col = 0
                row += 1

        total_rows = row + (1 if col > 0 else 0)
        return total_rows * cell_h

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_click(self, pos: tuple[int, int]) -> int | None:
        """Handle MOUSEBUTTONDOWN. Returns selected level_id or None."""
        for entry in self._rects:
            if entry["rect"].collidepoint(pos):
                if self.selected_id != entry["level_id"]:
                    self.selected_id = entry["level_id"]
                    audio_manager.play_sound_effect("button_click")
                return entry["level_id"]
        return None
