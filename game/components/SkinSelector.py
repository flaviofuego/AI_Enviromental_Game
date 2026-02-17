"""
Reusable Skin Selector component.
Provides a grid of selectable skin circles with preview and name labels.
"""
import pygame
import math


# Default available skins (shared across the game)
DEFAULT_SKINS = [
    {"id": "default", "name": "Predeterminado", "color": (200, 200, 200)},
    {"id": "eco_warrior", "name": "Guerrero Eco", "color": (100, 200, 100)},
    {"id": "arctic", "name": "Explorador Ártico", "color": (150, 220, 255)},
    {"id": "volcano", "name": "Resistente Volcánico", "color": (255, 100, 50)},
    {"id": "cyber", "name": "Hacker Climático", "color": (100, 255, 200)},
    {"id": "retro", "name": "Retro Salvador", "color": (255, 200, 100)},
    {"id": "scientist", "name": "Científico", "color": (200, 200, 255)},
    {"id": "agent", "name": "Agente Especial", "color": (50, 50, 100)},
]


class SkinSelector:
    """Renders a grid of selectable skins and a preview panel."""

    def __init__(self, skins: list[dict] = None, columns: int = 3,
                 cell_size: int = 70, padding: int = 25,
                 selected_id: str = "default"):
        self.skins = skins or list(DEFAULT_SKINS)
        self.columns = columns
        self.cell_size = cell_size
        self.padding = padding
        self.selected_id = selected_id

        self._font_name = pygame.font.Font(None, 14)
        self._font_preview_title = pygame.font.Font(None, 24)
        self._font_preview_label = pygame.font.Font(None, 18)

        # Cached rects for hit testing (populated after draw)
        self._skin_rects: list[dict] = []

    def get_selected_skin(self) -> dict:
        """Return the full dict of the currently selected skin."""
        for s in self.skins:
            if s["id"] == self.selected_id:
                return s
        return self.skins[0]

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_grid(self, surface: pygame.Surface, origin_x: int, origin_y: int,
                  animation_time: float = 0) -> list[dict]:
        """Draw the grid of selectable skins. Returns list of hit rects."""
        self._skin_rects = []
        cs = self.cell_size

        for i, skin in enumerate(self.skins):
            row = i // self.columns
            col = i % self.columns
            x = origin_x + col * (cs + self.padding + 20)
            y = origin_y + row * (cs + self.padding + 15)
            cx, cy = x + cs // 2, y + cs // 2
            r = cs // 2 - 3

            # Background circle
            pygame.draw.circle(surface, (40, 40, 70), (cx, cy), r + 5)
            # Skin color
            pygame.draw.circle(surface, skin["color"], (cx, cy), r)

            # Selection ring
            if skin["id"] == self.selected_id:
                gold = (255, 215, 0)
                pygame.draw.circle(surface, gold, (cx, cy), r + 2, 3)
                # Glow
                glow = pygame.Surface((cs + 10, cs + 10), pygame.SRCALPHA)
                pygame.draw.circle(glow, (*gold, 80), (cs // 2 + 5, cs // 2 + 5), r + 5)
                surface.blit(glow, (x - 5, y - 5))

            # Label
            name_surf = self._font_name.render(skin["name"], True, (255, 255, 255))
            name_rect = name_surf.get_rect(centerx=cx, top=y + cs + 5)
            surface.blit(name_surf, name_rect)

            self._skin_rects.append({
                "rect": pygame.Rect(x, y, cs, cs),
                "skin_id": skin["id"],
            })

        return self._skin_rects

    def draw_preview(self, surface: pygame.Surface,
                     x: int, y: int, width: int, height: int,
                     animation_time: float = 0):
        """Draw a large preview of the selected skin."""
        # Panel background
        pygame.draw.rect(surface, (30, 30, 60),
                         (x, y, width, height), border_radius=15)
        pygame.draw.rect(surface, (173, 216, 230),
                         (x, y, width, height), 2, border_radius=15)

        # Title
        title = self._font_preview_title.render("VISTA PREVIA", True, (255, 255, 255))
        surface.blit(title, (x + width // 2 - title.get_width() // 2, y + 15))

        skin = self.get_selected_skin()
        preview_r = min(width, height) // 3
        cx = x + width // 2
        cy = y + height // 2

        # Main circle
        skin_surf = pygame.Surface((preview_r * 2, preview_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(skin_surf, skin["color"], (preview_r, preview_r), preview_r)
        # Highlight
        pygame.draw.circle(skin_surf, (255, 255, 255, 150),
                           (preview_r, preview_r), preview_r // 2)
        surface.blit(skin_surf, (cx - preview_r, cy - preview_r))

        # Breathe animation ring
        pulse = abs(math.sin(animation_time * 2)) * 4
        pygame.draw.circle(surface, (*skin["color"][:3], 80),
                           (cx, cy), int(preview_r + 6 + pulse), 2)

        # Label
        label = self._font_preview_label.render(
            f"Skin: {skin['name']}", True, (255, 215, 0)
        )
        surface.blit(label, (cx - label.get_width() // 2,
                             cy + preview_r + 15))

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def handle_click(self, pos) -> str | None:
        """Check if *pos* clicked a skin.  Returns skin_id or None."""
        for entry in self._skin_rects:
            if entry["rect"].collidepoint(pos):
                self.selected_id = entry["skin_id"]
                return entry["skin_id"]
        return None
