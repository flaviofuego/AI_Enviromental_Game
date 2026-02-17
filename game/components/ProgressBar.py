"""
Reusable progress bar component for displaying percentage-based values.
"""
import pygame


class ProgressBar:
    """A horizontal progress bar with label, value text, and fill color."""

    def __init__(self, x: int, y: int, width: int, height: int = 8,
                 bg_color=(50, 50, 50),
                 fill_color=(173, 216, 230),
                 label: str = "",
                 label_font: pygame.font.Font = None,
                 label_color=(255, 255, 255),
                 value_color=None,
                 border_radius: int = 0):
        self.rect = pygame.Rect(x, y, width, height)
        self.bg_color = bg_color
        self.fill_color = fill_color
        self.label = label
        self.label_font = label_font or pygame.font.Font(None, 14)
        self.label_color = label_color
        self.value_color = value_color or fill_color
        self.border_radius = border_radius
        self._value: float = 0.0  # 0–100

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v: float):
        self._value = max(0.0, min(100.0, v))

    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> int:
        """Draw the progress bar.  Returns total height consumed
        (label + bar)."""
        y = self.rect.y
        consumed = 0

        # Label
        if self.label:
            lbl = self.label_font.render(self.label, True, self.label_color)
            surface.blit(lbl, (self.rect.x, y))

            val_text = f"{int(self._value)}%"
            val_surf = self.label_font.render(val_text, True, self.value_color)
            surface.blit(
                val_surf,
                (self.rect.right - val_surf.get_width(), y),
            )
            y += lbl.get_height() + 2
            consumed += lbl.get_height() + 2

        # Background
        bg_rect = pygame.Rect(self.rect.x, y, self.rect.width, self.rect.height)
        pygame.draw.rect(surface, self.bg_color, bg_rect,
                         border_radius=self.border_radius)

        # Fill
        fill_w = int((self._value / 100) * self.rect.width)
        if fill_w > 0:
            fill_rect = pygame.Rect(self.rect.x, y, fill_w, self.rect.height)
            pygame.draw.rect(surface, self.fill_color, fill_rect,
                             border_radius=self.border_radius)

        consumed += self.rect.height
        return consumed
