"""
Profile management sub-screens: list, create, and skin selection.
Extracted from home.py for component separation.
"""
import pygame
import time

from ..components.SkinSelector import SkinSelector, DEFAULT_SKINS
from ..components.Panel import Panel
from ..components.AudioManager import audio_manager
from ..components.FontCache import font_cache


class ProfileManager:
    """Handles the profile list, creation, and skin selection screens."""

    def __init__(self, screen: pygame.Surface, save_system, colors: dict):
        self.screen = screen
        self.sw = screen.get_width()
        self.sh = screen.get_height()
        self.save_system = save_system
        self.colors = colors
        self.is_mobile = self.sh > self.sw

        # Fonts (via FontCache)
        self.font_title = font_cache.get(None, 48 if not self.is_mobile else 36)
        self.font_subtitle = font_cache.get(None, 24 if not self.is_mobile else 20)
        self.font_text = font_cache.get(None, 18 if not self.is_mobile else 16)
        self.font_small = font_cache.get(None, 14 if not self.is_mobile else 12)

        # State
        self.current_screen = "list"  # list | create | skin
        self.profiles: list = []
        self.selected_index = -1
        self.input_text = ""
        self.input_active = False
        self.error_message = ""
        self.success_message = ""
        self.message_time = 0.0

        # Skin selector component
        self.skin_selector = SkinSelector(selected_id="default")

        # Cached UI rects from last draw — used by event handlers
        # to avoid double-rendering.
        self.cached_ui: dict | None = None

        self.load_profiles()

    # ------------------------------------------------------------------
    # Profile data
    # ------------------------------------------------------------------

    def load_profiles(self):
        self.profiles = self.save_system.get_all_profiles()
        if self.profiles:
            self.selected_index = 0

    def load_selected_profile(self) -> bool:
        if 0 <= self.selected_index < len(self.profiles):
            pid = self.profiles[self.selected_index]["profile_id"]
            profile = self.save_system.load_profile(pid)
            if profile:
                skin = profile.get("skin", "default")
                self.skin_selector.selected_id = skin
                audio_manager.load_audio_settings_from_profile(self.save_system)
                self.success_message = f"¡Bienvenido, {profile['player_name']}!"
                self.message_time = time.time()
                return True
            else:
                self.error_message = "Error al cargar el perfil."
                self.message_time = time.time()
        return False

    def create_new_profile(self) -> bool:
        name = self.input_text.strip()
        if not name:
            self.error_message = "Ingresa un nombre de jugador."
            self.message_time = time.time()
            return False
        pid = self.save_system.create_profile(name)
        if pid:
            profile = self.save_system.get_current_profile_data()
            profile["skin"] = "default"
            self.skin_selector.selected_id = "default"
            self.save_system.save_current_profile()
            audio_manager.save_audio_settings_to_profile(self.save_system)
            self.success_message = f"¡Perfil creado! Bienvenido, {name}."
            self.message_time = time.time()
            self.load_profiles()
            return True
        else:
            self.error_message = "Error al crear el perfil."
            self.message_time = time.time()
            return False

    def delete_selected_profile(self) -> bool:
        if 0 <= self.selected_index < len(self.profiles):
            pid = self.profiles[self.selected_index]["profile_id"]
            if self.save_system.delete_profile(pid):
                self.success_message = "Perfil eliminado."
                self.message_time = time.time()
                self.load_profiles()
                self.selected_index = 0 if self.profiles else -1
                return True
            else:
                self.error_message = "Error al eliminar."
                self.message_time = time.time()
        return False

    def save_skin(self, skin_id: str):
        if 0 <= self.selected_index < len(self.profiles):
            pid = self.profiles[self.selected_index]["profile_id"]
            profile = self.save_system.load_profile(pid)
            if profile:
                profile["skin"] = skin_id
                self.save_system.current_profile = profile
                self.save_system.save_current_profile()
                self.success_message = f"¡Skin guardada!"
                self.message_time = time.time()

    # ------------------------------------------------------------------
    # Drawing — Profile List
    # ------------------------------------------------------------------

    def draw_list(self, draw_bg_fn) -> dict:
        draw_bg_fn()

        pw = 600 if not self.is_mobile else self.sw - 40
        ph = 500 if not self.is_mobile else self.sh - 100
        px = (self.sw - pw) // 2
        py = (self.sh - ph) // 2

        panel = Panel(px, py, pw, ph,
                      bg_color=self.colors["panel_dark"],
                      border_color=self.colors["ice_blue"],
                      title="PERFILES DE JUGADOR",
                      title_font=self.font_title,
                      title_color=self.colors["ice_blue"])
        panel.draw(self.screen)

        ui = {"profiles": [], "create_button": None, "back_button": None,
              "load_button": None, "delete_button": None, "skin_button": None}

        # Profile list
        y_start = py + 100
        item_h = 60
        for i, prof in enumerate(self.profiles):
            y = y_start + i * (item_h + 10)
            is_sel = (i == self.selected_index)
            rect = pygame.Rect(px + 20, y, pw - 40, item_h)
            color = self.colors["button_hover"] if is_sel else (50, 50, 50, 180)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            border = self.colors["text_gold"] if is_sel else self.colors["ice_blue"]
            pygame.draw.rect(self.screen, border, rect, 2, border_radius=5)

            name = self.font_subtitle.render(prof["player_name"], True, self.colors["text_white"])
            self.screen.blit(name, (rect.x + 20, rect.y + 10))
            pts = self.font_text.render(f"Puntos: {prof['total_points']}", True, self.colors["text_gold"])
            self.screen.blit(pts, (rect.x + 20, rect.y + 35))
            date = self.font_small.render(f"Último acceso: {prof['last_played'][:10]}", True, self.colors["text_white"])
            self.screen.blit(date, (rect.x + pw // 2, rect.y + 38))

            ui["profiles"].append({"rect": rect, "index": i})

        # Action buttons
        bw, bh = 100, 30
        by = py + ph - 40
        sp = 10

        def _btn(x, color, text):
            r = pygame.Rect(x, by, bw, bh)
            pygame.draw.rect(self.screen, color, r, border_radius=3)
            t = self.font_small.render(text, True, self.colors["text_white"])
            self.screen.blit(t, t.get_rect(center=r.center))
            return r

        ui["create_button"] = _btn(px + 20, self.colors["hope_green"], "Nuevo")

        if self.selected_index >= 0:
            ui["load_button"] = _btn(px + 20 + bw + sp, self.colors["button_active"], "Cargar")
            ui["delete_button"] = _btn(px + 20 + (bw + sp) * 2, self.colors["critical_red"], "Eliminar")
            ui["skin_button"] = _btn(px + 20 + (bw + sp) * 3, self.colors.get("purple", (128, 0, 128)), "Skin")

        ui["back_button"] = _btn(px + pw - bw - 20, self.colors["warning_orange"], "Volver")

        self._draw_messages()
        self.cached_ui = ui
        return ui

    # ------------------------------------------------------------------
    # Drawing — Create Profile
    # ------------------------------------------------------------------

    def draw_create(self, draw_bg_fn, animation_time: float) -> dict:
        draw_bg_fn()

        pw, ph = 400, 300
        if self.is_mobile:
            pw = self.sw - 40
        px = (self.sw - pw) // 2
        py = (self.sh - ph) // 2

        panel = Panel(px, py, pw, ph,
                      bg_color=self.colors["panel_dark"],
                      border_color=self.colors["hope_green"],
                      title="CREAR NUEVO PERFIL",
                      title_font=self.font_subtitle,
                      title_color=self.colors["hope_green"])
        panel.draw(self.screen)

        # Instruction
        instr = self.font_small.render("Ingresa tu nombre de jugador", True, self.colors["text_white"])
        self.screen.blit(instr, instr.get_rect(centerx=self.sw // 2, top=py + 80))

        # Input field
        ir = pygame.Rect(px + 50, py + 100, pw - 100, 40)
        border_c = self.colors["text_white"] if self.input_active else (100, 100, 100)
        pygame.draw.rect(self.screen, border_c, ir, 2)
        ts = self.font_text.render(self.input_text, True, self.colors["text_white"])
        self.screen.blit(ts, (ir.x + 10, ir.y + 10))
        if self.input_active and int(animation_time * 2) % 2:
            cx = ir.x + 10 + ts.get_width()
            pygame.draw.line(self.screen, self.colors["text_white"],
                             (cx, ir.y + 5), (cx, ir.y + 35), 2)

        # Buttons
        by = py + ph - 60
        cr = pygame.Rect(px + 50, by, 100, 40)
        pygame.draw.rect(self.screen, self.colors["hope_green"], cr, border_radius=5)
        ct = self.font_text.render("Crear", True, self.colors["text_white"])
        self.screen.blit(ct, ct.get_rect(center=cr.center))

        ca = pygame.Rect(px + pw - 150, by, 100, 40)
        pygame.draw.rect(self.screen, self.colors["critical_red"], ca, border_radius=5)
        cat = self.font_text.render("Cancelar", True, self.colors["text_white"])
        self.screen.blit(cat, cat.get_rect(center=ca.center))

        self._draw_messages()
        ui = {"input_rect": ir, "create_button": cr, "cancel_button": ca}
        self.cached_ui = ui
        return ui

    # ------------------------------------------------------------------
    # Drawing — Skin Selection
    # ------------------------------------------------------------------

    def draw_skin_screen(self, draw_bg_fn, animation_time: float) -> dict:
        draw_bg_fn()

        pw = 700 if not self.is_mobile else self.sw - 40
        ph = 550 if not self.is_mobile else self.sh - 100
        px = (self.sw - pw) // 2
        py = (self.sh - ph) // 2

        panel = Panel(px, py, pw, ph,
                      bg_color=(20, 20, 40, 220),
                      border_color=self.colors.get("purple", (128, 0, 128)),
                      border_width=3, border_radius=10,
                      title="SELECCIONAR SKIN",
                      title_font=self.font_title,
                      title_color=self.colors["ice_blue"])
        panel.draw(self.screen, glow=True, animation_time=animation_time)

        # Profile name
        if 0 <= self.selected_index < len(self.profiles):
            pn = self.profiles[self.selected_index]["player_name"]
            sub = self.font_subtitle.render(f"Perfil: {pn}", True, self.colors["text_white"])
            self.screen.blit(sub, sub.get_rect(centerx=self.sw // 2, top=py + 90))

        # Load current skin from profile
        if self.save_system.current_profile:
            self.skin_selector.selected_id = self.save_system.current_profile.get(
                "skin", self.skin_selector.selected_id)

        # Grid (left side)
        grid_rects = self.skin_selector.draw_grid(
            self.screen, px + 25, py + 140, animation_time)

        # Preview (right side)
        prev_w = 250
        prev_x = px + pw - prev_w - 30
        prev_y = py + 140
        prev_h = ph - 200
        self.skin_selector.draw_preview(
            self.screen, prev_x, prev_y, prev_w, prev_h, animation_time)

        # Back button
        bw, bh = 100, 30
        bb = pygame.Rect(px + pw - bw - 30, py + ph - bh - 20, bw, bh)
        pygame.draw.rect(self.screen, self.colors["warning_orange"], bb, border_radius=5)
        pygame.draw.rect(self.screen, (255, 180, 50), bb, 2, border_radius=5)
        bt = self.font_text.render("VOLVER", True, self.colors["text_white"])
        self.screen.blit(bt, bt.get_rect(center=bb.center))

        self._draw_messages()
        ui = {"skins": grid_rects, "back_button": bb}
        self.cached_ui = ui
        return ui

    # ------------------------------------------------------------------
    # Event Handling
    # ------------------------------------------------------------------

    def handle_list_events(self, event, ui):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            for p in ui["profiles"]:
                if p["rect"].collidepoint(pos):
                    self.selected_index = p["index"]
                    return None
            if ui["create_button"] and ui["create_button"].collidepoint(pos):
                self.current_screen = "create"
                self.input_text = ""
                self.input_active = True
                return None
            if ui["back_button"] and ui["back_button"].collidepoint(pos):
                return "back"
            if ui.get("load_button") and ui["load_button"].collidepoint(pos):
                if self.load_selected_profile():
                    return "back"
            if ui.get("delete_button") and ui["delete_button"].collidepoint(pos):
                self.delete_selected_profile()
            if ui.get("skin_button") and ui["skin_button"].collidepoint(pos):
                self.current_screen = "skin"
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return "back"
        return None

    def handle_create_events(self, event, ui):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            self.input_active = ui["input_rect"].collidepoint(pos)
            if ui["create_button"].collidepoint(pos):
                if self.create_new_profile():
                    return "back"
            if ui["cancel_button"].collidepoint(pos):
                self.current_screen = "list"
        elif event.type == pygame.KEYDOWN:
            if self.input_active:
                if event.key == pygame.K_RETURN:
                    if self.create_new_profile():
                        return "back"
                elif event.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    self.current_screen = "list"
                    self.input_text = ""
                elif len(self.input_text) < 20:
                    self.input_text += event.unicode
        return None

    def handle_skin_events(self, event, ui):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            clicked_id = self.skin_selector.handle_click(pos)
            if clicked_id:
                self.save_skin(clicked_id)
                return None
            if ui["back_button"].collidepoint(pos):
                self.current_screen = "list"
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.current_screen = "list"
        return None

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def _draw_messages(self):
        now = time.time()
        if self.error_message and (now - self.message_time) < 3:
            self._blit_msg(self.error_message, self.colors["critical_red"], (40, 0, 0))
        if self.success_message and (now - self.message_time) < 3:
            self._blit_msg(self.success_message, self.colors["hope_green"], (0, 40, 0))

    def _blit_msg(self, text, color, bg_tint):
        surf = self.font_text.render(text, True, color)
        rect = surf.get_rect(center=(self.sw // 2, self.sh - 100))
        bg = pygame.Rect(rect.x - 10, rect.y - 5, rect.width + 20, rect.height + 10)
        pygame.draw.rect(self.screen, bg_tint, bg, border_radius=5)
        pygame.draw.rect(self.screen, color, bg, 2, border_radius=5)
        self.screen.blit(surf, rect)
