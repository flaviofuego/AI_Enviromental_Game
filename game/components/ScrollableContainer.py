"""
Scrollable container that clips its children to a fixed viewport.
Useful for long lists of items that exceed available screen space.
"""
import pygame


class ScrollableContainer:
    """A scrollable viewport that clips content drawn inside it."""

    def __init__(self, rect: pygame.Rect, scroll_speed: int = 20):
        self.rect = rect
        self.scroll_speed = scroll_speed
        self.scroll_offset = 0
        self.content_height = 0

    @property
    def max_scroll(self) -> int:
        return max(0, self.content_height - self.rect.height)

    @property
    def needs_scroll(self) -> bool:
        return self.content_height > self.rect.height

    # ------------------------------------------------------------------
    # Scrolling helpers
    # ------------------------------------------------------------------

    def scroll_up(self):
        self.scroll_offset = max(0, self.scroll_offset - self.scroll_speed)

    def scroll_down(self):
        self.scroll_offset = min(self.max_scroll,
                                 self.scroll_offset + self.scroll_speed)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle scroll events.  Returns True if consumed."""
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            if event.button == 4:
                self.scroll_up()
                return True
            elif event.button == 5:
                self.scroll_down()
                return True
        return False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def begin(self, surface: pygame.Surface) -> pygame.Surface:
        """Create a content surface to draw on.  Call :meth:`end` after."""
        self._parent = surface
        h = max(self.rect.height, self.content_height)
        self._content_surf = pygame.Surface(
            (self.rect.width, h), pygame.SRCALPHA
        )
        return self._content_surf

    def end(self):
        """Blit the visible portion of content onto the parent surface."""
        visible = pygame.Rect(0, self.scroll_offset,
                              self.rect.width, self.rect.height)
        self._parent.blit(self._content_surf, self.rect.topleft, visible)

        # scroll indicators
        if self.needs_scroll:
            if self.scroll_offset > 0:
                pygame.draw.polygon(
                    self._parent, (200, 200, 200),
                    [
                        (self.rect.right - 20, self.rect.y + 8),
                        (self.rect.right - 15, self.rect.y + 3),
                        (self.rect.right - 10, self.rect.y + 8),
                    ],
                )
            if self.scroll_offset < self.max_scroll:
                by = self.rect.bottom
                pygame.draw.polygon(
                    self._parent, (200, 200, 200),
                    [
                        (self.rect.right - 20, by - 8),
                        (self.rect.right - 15, by - 3),
                        (self.rect.right - 10, by - 8),
                    ],
                )
