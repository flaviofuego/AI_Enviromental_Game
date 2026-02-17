"""
Reusable text rendering component with word-wrapping, clipping, and scrolling support.
Eliminates text overflow issues in modals and panels.
"""
import pygame


class TextRenderer:
    """Renders text with automatic word-wrapping and overflow protection."""

    def __init__(self, font: pygame.font.Font = None, color=(255, 255, 255),
                 line_spacing: int = 4):
        self.font = font or pygame.font.Font(None, 20)
        self.color = color
        self.line_spacing = line_spacing

    # ------------------------------------------------------------------
    # Word-wrap helpers
    # ------------------------------------------------------------------

    def wrap_text(self, text: str, max_width: int) -> list[str]:
        """Split *text* into lines that fit within *max_width* pixels."""
        words = text.split(" ")
        lines: list[str] = []
        current: list[str] = []

        for word in words:
            test_line = " ".join(current + [word])
            if self.font.size(test_line)[0] <= max_width:
                current.append(word)
            else:
                if current:
                    lines.append(" ".join(current))
                current = [word]

        if current:
            lines.append(" ".join(current))
        return lines

    def wrap_multiline(self, text: str, max_width: int) -> list[str]:
        """Handle text that may already contain ``\\n`` characters."""
        result: list[str] = []
        for paragraph in text.split("\n"):
            if paragraph.strip() == "":
                result.append("")
            else:
                result.extend(self.wrap_text(paragraph, max_width))
        return result

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_lines(self, surface: pygame.Surface, lines: list[str],
                     x: int, y: int, max_height: int = 0,
                     color=None) -> int:
        """Blit pre-wrapped *lines* onto *surface* starting at (*x*, *y*).

        If *max_height* > 0 the text is clipped so it never exceeds the
        bounding box.  Returns the total height consumed.
        """
        col = color or self.color
        line_h = self.font.get_linesize() + self.line_spacing
        drawn = 0

        for line in lines:
            if max_height > 0 and (drawn + line_h) > max_height:
                break
            if line.strip():
                surf = self.font.render(line, True, col)
                surface.blit(surf, (x, y + drawn))
            drawn += line_h
        return drawn

    def render_text(self, surface: pygame.Surface, text: str,
                    x: int, y: int, max_width: int,
                    max_height: int = 0, color=None) -> int:
        """Wrap *text* and render it.  Returns consumed height."""
        lines = self.wrap_multiline(text, max_width)
        return self.render_lines(surface, lines, x, y,
                                 max_height=max_height, color=color)

    def get_text_height(self, text: str, max_width: int) -> int:
        """Calculate height that *text* would occupy after wrapping."""
        lines = self.wrap_multiline(text, max_width)
        line_h = self.font.get_linesize() + self.line_spacing
        return len(lines) * line_h
