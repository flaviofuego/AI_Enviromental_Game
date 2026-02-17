"""
PvP (Player vs Player) game setup page.
Allows configuring score limit, time limit, powerups, background level,
and player skins before launching a local 2-player match.

Components used:
- Panel              -- section containers
- OptionSelector     -- arrow-based option picker (score, time)
- BackgroundSelector -- level-background thumbnail picker
- SkinSelector       -- skin grid for each player
- IconRenderer       -- drawn icons (no emoji)
- AudioManager       -- sound effects
"""
import pygame
import time

from ..components.Panel import Panel
from ..components.SkinSelector import SkinSelector
from ..components.OptionSelector import OptionSelector
from ..components.BackgroundSelector import BackgroundSelector
from ..components.IconRenderer import IconRenderer
from ..components.AudioManager import audio_manager
from ..components.FontCache import font_cache
from ..components.GameButton import text_button


class PvPSetupScreen:
    """Setup screen for a local Player vs Player match."""

    # Available score limits (plan: 1, 2, 3, 5, 7, 10)
    SCORE_OPTIONS = [1, 2, 3, 5, 7, 10]
    SCORE_LABELS = ["1", "2", "3", "5", "7", "10"]

    # Available time limits (seconds, 0 = unlimited) -- (plan: 0, 300, 600, 900)
    TIME_OPTIONS = [0, 300, 600, 900]
    TIME_LABELS = ["Sin limite", "5 min", "10 min", "15 min"]

    def __init__(self, screen: pygame.Surface, save_system=None):
        self.screen = screen
        self.sw = screen.get_width()
        self.sh = screen.get_height()
        self.save_system = save_system
        self.is_mobile = self.sh > self.sw
        self.clock = pygame.time.Clock()
        self.animation_time = 0.0

        # ------------------------------------------------------------------
        # Fonts (via FontCache)
        # ------------------------------------------------------------------
        self.font_title = font_cache.get(None, 42 if not self.is_mobile else 32)
        self.font_subtitle = font_cache.get(None, 24 if not self.is_mobile else 20)
        self.font_text = font_cache.get(None, 20 if not self.is_mobile else 18)
        self.font_small = font_cache.get(None, 16 if not self.is_mobile else 14)

        # ------------------------------------------------------------------
        # Colour palette
        # ------------------------------------------------------------------
        self.colors = {
            "bg": (15, 15, 35),
            "panel": (20, 20, 40, 220),
            "ice_blue": (173, 216, 230),
            "gold": (255, 215, 0),
            "green": (34, 139, 34),
            "red": (220, 50, 50),
            "orange": (255, 140, 0),
            "white": (255, 255, 255),
            "button": (0, 100, 200),
            "button_hover": (0, 150, 255),
            "purple": (128, 0, 128),
            "dark_panel": (30, 30, 60),
        }

        # ------------------------------------------------------------------
        # Sub-components
        # ------------------------------------------------------------------
        self.score_selector = OptionSelector(
            label="Goles para ganar:",
            options=self.SCORE_OPTIONS,
            labels=self.SCORE_LABELS,
            selected_index=2,  # default: 3
            label_font=self.font_text,
            value_font=self.font_subtitle,
            label_color=self.colors["white"],
            value_color=self.colors["gold"],
            arrow_bg=self.colors["button"],
            arrow_fg=self.colors["white"],
            value_width=80,
        )

        self.time_selector = OptionSelector(
            label="Limite de tiempo:",
            options=self.TIME_OPTIONS,
            labels=self.TIME_LABELS,
            selected_index=0,  # default: unlimited
            label_font=self.font_text,
            value_font=self.font_subtitle,
            label_color=self.colors["white"],
            value_color=self.colors["gold"],
            arrow_bg=self.colors["button"],
            arrow_fg=self.colors["white"],
            value_width=120,
        )

        self.bg_selector = BackgroundSelector(
            thumb_size=(100, 58),
            spacing=8,
            selected_id=1,
            label_font=self.font_small,
        )

        self.p1_skin_selector = SkinSelector(
            columns=4, cell_size=50, padding=12, selected_id="default")
        self.p2_skin_selector = SkinSelector(
            columns=4, cell_size=50, padding=12, selected_id="eco_warrior")

        self.powerups_enabled = True
        self.expanded_skin = 0  # 0=none, 1=p1, 2=p2

        # ------------------------------------------------------------------
        # Transient message
        # ------------------------------------------------------------------
        self.message = ""
        self.message_time = 0.0
        self.message_type = "info"

        # ------------------------------------------------------------------
        # Cached hit rects (populated during draw for layout-dependent btns)
        # ------------------------------------------------------------------
        self._powerup_rect = pygame.Rect(0, 0, 0, 0)
        self._p1_change_rect = pygame.Rect(0, 0, 0, 0)
        self._p2_change_rect = pygame.Rect(0, 0, 0, 0)

        # ------------------------------------------------------------------
        # Fixed-position buttons (GameButton)
        # ------------------------------------------------------------------
        bar_y = self.sh - 60
        bw, bh = 150, 40

        self.btn_back = text_button(
            text="Volver",
            position=(20, bar_y),
            size=(bw, bh),
            bg_color=self.colors["orange"],
            hover_color=(255, 180, 40),
            border_radius=6,
            font_size=20 if not self.is_mobile else 18,
            icon_draw_fn=IconRenderer.draw_arrow_left,
            icon_padding=6,
        )
        self.btn_play = text_button(
            text="JUGAR",
            position=(self.sw - bw - 20, bar_y),
            size=(bw, bh),
            bg_color=self.colors["green"],
            hover_color=(50, 180, 50),
            border_radius=6,
            font_size=24 if not self.is_mobile else 20,
            icon_draw_fn=IconRenderer.draw_play,
            icon_padding=4,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def score_limit(self) -> int:
        return self.score_selector.value

    @property
    def time_limit(self) -> float | None:
        v = self.time_selector.value
        return v if v > 0 else None

    @property
    def time_label(self) -> str:
        return self.time_selector.display_label

    @property
    def background_level_id(self) -> int:
        return self.bg_selector.selected_id

    def _show_msg(self, msg, mtype="info"):
        self.message = msg
        self.message_type = mtype
        self.message_time = time.time()

    # ==================================================================
    # Drawing
    # ==================================================================

    def draw(self):
        self.screen.fill(self.colors["bg"])

        # ---- Title ----
        title = self.font_title.render(
            "PARTIDA LOCAL - 2 JUGADORES", True, self.colors["ice_blue"])
        shadow = self.font_title.render(
            "PARTIDA LOCAL - 2 JUGADORES", True, (0, 0, 0))
        cx = self.sw // 2
        tr = title.get_rect(centerx=cx, top=30)
        self.screen.blit(shadow, (tr.x + 2, tr.y + 2))
        self.screen.blit(title, tr)

        sub = self.font_subtitle.render(
            "Configura la partida y elige tus skins", True, self.colors["gold"])
        self.screen.blit(sub, sub.get_rect(centerx=cx, top=tr.bottom + 5))

        # ---- Two-column layout ----
        col_w = (self.sw - 60) // 2
        left_x = 20
        right_x = left_x + col_w + 20
        section_y = 100

        self._draw_match_settings(left_x, section_y, col_w)
        self._draw_player_skins(right_x, section_y, col_w)
        self._draw_bottom_bar()
        self._draw_message()

    # ------------------------------------------------------------------
    # Left column: match settings + background selector
    # ------------------------------------------------------------------

    def _draw_match_settings(self, x, y, w):
        panel = Panel(x, y, w, self.sh - 180,
                      bg_color=self.colors["panel"],
                      border_color=self.colors["ice_blue"],
                      border_radius=8,
                      title="CONFIGURACION",
                      title_font=self.font_subtitle,
                      title_color=self.colors["ice_blue"])
        panel.draw(self.screen)

        cr = panel.content_rect
        cy = cr.y + 10

        # Score selector
        h = self.score_selector.draw(self.screen, cr.x, cy)
        cy += h + 12

        # Time selector
        h = self.time_selector.draw(self.screen, cr.x, cy)
        cy += h + 12

        # Power-ups toggle
        lbl = self.font_text.render("Power-ups:", True, self.colors["white"])
        self.screen.blit(lbl, (cr.x, cy))

        self._powerup_rect = pygame.Rect(cr.x + 120, cy - 2, 100, 28)
        pu_color = self.colors["green"] if self.powerups_enabled else self.colors["red"]
        pu_text = "Activados" if self.powerups_enabled else "Desactivados"
        pygame.draw.rect(self.screen, pu_color, self._powerup_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.colors["white"], self._powerup_rect, 1, border_radius=4)
        pt = self.font_small.render(pu_text, True, self.colors["white"])
        self.screen.blit(pt, pt.get_rect(center=self._powerup_rect.center))
        cy += 40

        # ---- Background / level selector ----
        pygame.draw.line(self.screen, self.colors["ice_blue"],
                         (cr.x, cy), (cr.x + cr.width - 10, cy))
        cy += 10
        bg_label = self.font_text.render("Escenario:", True, self.colors["ice_blue"])
        self.screen.blit(bg_label, (cr.x, cy))
        cy += bg_label.get_height() + 8

        self.bg_selector.draw(self.screen, cr.x, cy, max_width=cr.width)
        cy += 90  # approx height of thumbnails

        # ---- Summary ----
        pygame.draw.line(self.screen, self.colors["ice_blue"],
                         (cr.x, cy), (cr.x + cr.width - 10, cy))
        cy += 10
        summary = self.font_text.render("Resumen:", True, self.colors["ice_blue"])
        self.screen.blit(summary, (cr.x, cy))
        cy += summary.get_height() + 8

        from ..config.level_config import LEVELS
        level_name = LEVELS.get(self.background_level_id, {}).get("name", "?")
        details = [
            f"  Goles: Primero a {self.score_limit}",
            f"  Tiempo: {self.time_label}",
            f"  Power-ups: {'Si' if self.powerups_enabled else 'No'}",
            f"  Escenario: {level_name}",
        ]
        for d in details:
            ds = self.font_small.render(d, True, self.colors["white"])
            self.screen.blit(ds, (cr.x + 10, cy))
            cy += ds.get_height() + 4

    # ------------------------------------------------------------------
    # Right column: player skins
    # ------------------------------------------------------------------

    def _draw_player_skins(self, x, y, w):
        panel = Panel(x, y, w, self.sh - 180,
                      bg_color=self.colors["panel"],
                      border_color=self.colors["purple"],
                      border_radius=8,
                      title="JUGADORES",
                      title_font=self.font_subtitle,
                      title_color=self.colors["purple"])
        panel.draw(self.screen)

        cr = panel.content_rect
        cy = cr.y + 5
        half_h = (cr.height - 30) // 2

        self._draw_player_skin_section(
            cr.x, cy, cr.width, half_h,
            "Jugador 1 (WASD)", self.p1_skin_selector, 1)

        cy += half_h + 15
        pygame.draw.line(self.screen, (80, 80, 120),
                         (cr.x, cy - 8), (cr.x + cr.width, cy - 8))

        self._draw_player_skin_section(
            cr.x, cy, cr.width, half_h,
            "Jugador 2 (Flechas)", self.p2_skin_selector, 2)

    def _draw_player_skin_section(self, x, y, w, h,
                                   label: str, selector: SkinSelector,
                                   player_num: int):
        lbl = self.font_text.render(label, True, self.colors["gold"])
        self.screen.blit(lbl, (x, y))

        skin = selector.get_selected_skin()

        # Mini preview circle + name
        preview_r = 20
        px = x + w - preview_r - 10
        py_c = y + 10
        pygame.draw.circle(self.screen, (40, 40, 70), (px, py_c), preview_r + 3)
        pygame.draw.circle(self.screen, skin["color"], (px, py_c), preview_r)
        pygame.draw.circle(self.screen, (255, 255, 255, 150), (px, py_c), preview_r // 2)

        val = self.font_small.render(skin["name"], True, self.colors["white"])
        self.screen.blit(val, val.get_rect(right=px - preview_r - 8, centery=py_c))

        # Expand / collapse button  (drawn icon arrows instead of emoji)
        btn_rect = pygame.Rect(x, y + lbl.get_height() + 6, 100, 24)
        if player_num == 1:
            self._p1_change_rect = btn_rect
        else:
            self._p2_change_rect = btn_rect

        is_expanded = self.expanded_skin == player_num
        btn_col = self.colors["button_hover"] if is_expanded else self.colors["button"]
        pygame.draw.rect(self.screen, btn_col, btn_rect, border_radius=4)

        # Text + arrow icon
        btn_label = "Cerrar" if is_expanded else "Cambiar"
        bt = self.font_small.render(btn_label, True, self.colors["white"])
        text_x = btn_rect.x + 8
        self.screen.blit(bt, (text_x, btn_rect.centery - bt.get_height() // 2))

        arrow_rect = pygame.Rect(btn_rect.right - 22, btn_rect.y + 2, 20, 20)
        if is_expanded:
            IconRenderer.draw_arrow_up(self.screen, arrow_rect, self.colors["white"], padding=5)
        else:
            IconRenderer.draw_arrow_down(self.screen, arrow_rect, self.colors["white"], padding=5)

        # Expanded grid
        if is_expanded:
            grid_y = y + lbl.get_height() + 36
            selector.draw_grid(self.screen, x, grid_y, self.animation_time)

    # ------------------------------------------------------------------
    # Bottom bar
    # ------------------------------------------------------------------

    def _draw_bottom_bar(self):
        self.btn_back.draw(self.screen)
        self.btn_play.draw(self.screen)

    # ------------------------------------------------------------------
    # Message toast
    # ------------------------------------------------------------------

    def _draw_message(self):
        if not self.message or (time.time() - self.message_time) > 3:
            return
        colors_map = {
            "error": self.colors["red"],
            "success": self.colors["green"],
            "info": self.colors["ice_blue"],
        }
        col = colors_map.get(self.message_type, self.colors["ice_blue"])
        surf = self.font_text.render(self.message, True, col)
        rect = surf.get_rect(centerx=self.sw // 2, bottom=self.sh - 70)
        bg = rect.inflate(20, 10)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), bg, border_radius=5)
        pygame.draw.rect(self.screen, col, bg, 2, border_radius=5)
        self.screen.blit(surf, rect)

    # ==================================================================
    # Event handling
    # ==================================================================

    def handle_event(self, event) -> str | None:
        """Process a single event. Returns action string or None."""
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return "back"

        # Dispatch to GameButton instances (handles all event types)
        if self.btn_back.update(event):
            return "back"
        if self.btn_play.update(event):
            return "play"

        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return None

        pos = event.pos

        # Delegate to sub-components
        if self.score_selector.handle_click(pos):
            return None
        if self.time_selector.handle_click(pos):
            return None

        # Background selector
        if self.bg_selector.handle_click(pos) is not None:
            return None

        # Powerups toggle
        if self._powerup_rect.collidepoint(pos):
            audio_manager.play_sound_effect("button_click")
            self.powerups_enabled = not self.powerups_enabled
            return None

        # Skin expand/collapse
        if self._p1_change_rect.collidepoint(pos):
            audio_manager.play_sound_effect("button_click")
            self.expanded_skin = 0 if self.expanded_skin == 1 else 1
            return None
        if self._p2_change_rect.collidepoint(pos):
            audio_manager.play_sound_effect("button_click")
            self.expanded_skin = 0 if self.expanded_skin == 2 else 2
            return None

        # Skin grid clicks
        if self.expanded_skin == 1:
            if self.p1_skin_selector.handle_click(pos):
                audio_manager.play_sound_effect("button_click")
        elif self.expanded_skin == 2:
            if self.p2_skin_selector.handle_click(pos):
                audio_manager.play_sound_effect("button_click")

        return None

    # ==================================================================
    # Config output
    # ==================================================================

    def get_config(self) -> dict:
        """Return the match configuration dict."""
        return {
            "score_limit": self.score_limit,
            "time_limit": self.time_limit,
            "powerups": self.powerups_enabled,
            "p1_skin": self.p1_skin_selector.selected_id,
            "p2_skin": self.p2_skin_selector.selected_id,
            "background_level_id": self.background_level_id,
        }

    # ==================================================================
    # Main loop
    # ==================================================================

    def run(self) -> str | dict | None:
        """Run the setup screen.  Returns 'back', config dict, or 'exit'."""
        while True:
            dt = self.clock.tick(60) / 1000.0
            self.animation_time += dt

            for event in pygame.event.get():
                audio_manager.process_event(event)

                if event.type == pygame.QUIT:
                    return "exit"
                action = self.handle_event(event)
                if action == "back":
                    return "back"
                elif action == "play":
                    return self.get_config()

            # Advance hover animations
            self.btn_back.update_animation(dt)
            self.btn_play.update_animation(dt)

            self.draw()
            pygame.display.flip()
