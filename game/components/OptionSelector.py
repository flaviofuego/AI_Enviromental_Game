"""
Reusable left/right option selector component.
Renders a label, left arrow, current value, and right arrow.
Handles click events on arrows to cycle through options.
Separates update (event handling) from draw (rendering).
"""
import pygame
from .IconRenderer import IconRenderer
from .AudioManager import audio_manager


class OptionSelector:
    """A horizontal [<] value [>] selector that cycles through a list of options.

    Parameters
    ----------
    label : str
        Text displayed above the selector row.
    options : list
        The raw values to cycle through (e.g. ``[1, 2, 3, 5]``).
    labels : list[str] | None
        Human-readable labels per option.  If *None*, ``str(option)`` is used.
    selected_index : int
        Initial selection index.
    label_font : pygame.font.Font
        Font used for the label text.
    value_font : pygame.font.Font
        Font used for the current value.
    label_color : tuple
        RGB color for the label.
    value_color : tuple
        RGB color for the value text.
    arrow_bg : tuple
        Background color of arrow buttons.
    arrow_fg : tuple
        Foreground (icon) color of arrow buttons.
    arrow_size : int
        Width and height of each arrow button (square).
    value_width : int
        Horizontal space reserved for the value text between arrows.
    """

    def __init__(
        self,
        label: str,
        options: list,
        labels: list[str] | None = None,
        selected_index: int = 0,
        label_font: pygame.font.Font | None = None,
        value_font: pygame.font.Font | None = None,
        label_color: tuple = (255, 255, 255),
        value_color: tuple = (255, 215, 0),
        arrow_bg: tuple = (0, 100, 200),
        arrow_fg: tuple = (255, 255, 255),
        arrow_size: int = 30,
        value_width: int = 100,
    ):
        self.label = label
        self.options = list(options)
        self.labels = labels or [str(o) for o in self.options]
        self._index = max(0, min(selected_index, len(self.options) - 1))

        self.label_font = label_font or pygame.font.Font(None, 20)
        self.value_font = value_font or pygame.font.Font(None, 24)
        self.label_color = label_color
        self.value_color = value_color
        self.arrow_bg = arrow_bg
        self.arrow_fg = arrow_fg
        self.arrow_size = arrow_size
        self.value_width = value_width

        # Rects populated by draw() — used for hit-testing
        self._left_rect = pygame.Rect(0, 0, 0, 0)
        self._right_rect = pygame.Rect(0, 0, 0, 0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, val: int):
        self._index = max(0, min(val, len(self.options) - 1))

    @property
    def value(self):
        """The currently selected raw value."""
        return self.options[self._index]

    @property
    def display_label(self) -> str:
        """The human-readable label for the current selection."""
        return self.labels[self._index]

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface, x: int, y: int) -> int:
        """Render at (*x*, *y*). Returns the total height consumed."""
        # Label
        lbl_surf = self.label_font.render(self.label, True, self.label_color)
        surface.blit(lbl_surf, (x, y))
        row_y = y + lbl_surf.get_height() + 6

        sz = self.arrow_size

        # Left arrow
        self._left_rect = pygame.Rect(x, row_y, sz, sz)
        pygame.draw.rect(surface, self.arrow_bg, self._left_rect, border_radius=4)
        pygame.draw.rect(surface, (255, 255, 255), self._left_rect, 1, border_radius=4)
        IconRenderer.draw_arrow_left(surface, self._left_rect, self.arrow_fg, padding=10)

        # Value text centered between arrows
        val_center_x = x + sz + self.value_width // 2
        val_surf = self.value_font.render(self.display_label, True, self.value_color)
        surface.blit(val_surf, val_surf.get_rect(centerx=val_center_x, centery=row_y + sz // 2))

        # Right arrow
        right_x = x + sz + self.value_width
        self._right_rect = pygame.Rect(right_x, row_y, sz, sz)
        pygame.draw.rect(surface, self.arrow_bg, self._right_rect, border_radius=4)
        pygame.draw.rect(surface, (255, 255, 255), self._right_rect, 1, border_radius=4)
        IconRenderer.draw_arrow_right(surface, self._right_rect, self.arrow_fg, padding=10)

        return (row_y + sz) - y + 4

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_click(self, pos: tuple[int, int]) -> bool:
        """Handle a MOUSEBUTTONDOWN at *pos*. Returns True if selection changed."""
        if self._left_rect.collidepoint(pos):
            if self._index > 0:
                self._index -= 1
                audio_manager.play_sound_effect("button_click")
                return True
        elif self._right_rect.collidepoint(pos):
            if self._index < len(self.options) - 1:
                self._index += 1
                audio_manager.play_sound_effect("button_click")
                return True
        return False
