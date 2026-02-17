"""
Main menu / hub screen for Hockey Is Melting Down.
Refactored: environmental effects, profile management, and skin selection
live in their own components.  This file now focuses only on the main-menu
drawing and navigation loop.
"""
import pygame
import math
import time
import os

from ..config.save_system import GameSaveSystem
from ..components.GameButton import GameButton, image_button
from ..components.PopUp import PopUp
from ..components.modals import create_help_popup, create_settings_popup
from ..components.AudioManager import audio_manager
from ..components.EnvironmentalEffects import EnvironmentalEffects
from ..components.Panel import Panel
from ..components.ProgressBar import ProgressBar
from ..components.TextRenderer import TextRenderer
from ..components.FontCache import font_cache
from .profile_manager import ProfileManager


class HockeyMainScreen:
    def __init__(self, screen=None, save_system=None):
        # Screen setup
        if screen is None:
            pygame.init()
            info = pygame.display.Info()
            self.screen_width = min(1200, info.current_w - 100)
            self.screen_height = min(800, info.current_h - 100)
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Hockey Is Melting Down - Salva la Tierra")
        else:
            self.screen = screen
            self.screen_width = screen.get_width()
            self.screen_height = screen.get_height()

        self.is_mobile = self.screen_height > self.screen_width

        # Background image
        self.background_image = pygame.image.load("game/assets/background.png")
        self.background_opacity = 200

        # Save system
        self.save_system = save_system if save_system else GameSaveSystem()

        # Colors
        self.colors = {
            "bg_gradient_top": (173, 216, 230),
            "bg_gradient_bottom": (120, 50, 50),
            "ice_blue": (173, 216, 230),
            "critical_red": (220, 50, 50),
            "hope_green": (34, 139, 34),
            "warning_orange": (255, 140, 0),
            "text_white": (255, 255, 255),
            "text_gold": (255, 215, 0),
            "panel_dark": (20, 20, 40, 200),
            "button_active": (0, 100, 200),
            "button_hover": (0, 150, 255),
            "purple": (128, 0, 128),
        }

        # Fonts (via FontCache — no per-frame allocation)
        self.font_title = font_cache.get(None, 48 if not self.is_mobile else 36)
        self.font_subtitle = font_cache.get(None, 24 if not self.is_mobile else 20)
        self.font_text = font_cache.get(None, 18 if not self.is_mobile else 16)
        self.font_small = font_cache.get(None, 14 if not self.is_mobile else 12)

        # Text renderer for wrapping
        self.text_renderer = TextRenderer(self.font_text, self.colors["text_white"])

        # Environmental effects (extracted component)
        self.env_effects = EnvironmentalEffects(
            self.screen_width, self.screen_height, self.colors
        )

        # State
        self.current_screen = "main"  # main | profiles
        self.animation_time = 0.0

        # Game data
        self.game_data = {
            "player_points": 0,
            "planetary_progress": {
                "oceanos_limpiados": 0,
                "ozono_restaurado": 0,
                "aire_purificado": 0,
                "bosques_replantados": 0,
                "ciudades_enfriadas": 0,
            },
            "levels_unlocked": 1,
            "current_level": 1,
        }

        # Enemy agents (for GAIA panel)
        self.enemy_agents = [
            {"name": "SLICKWAVE", "desc": "Emperador del plástico\nInunda los océanos con desechos", "unlocked": True, "defeated": False},
            {"name": "UVBLADE", "desc": "Destructor del ozono\nHa perforado el escudo celestial", "unlocked": True, "defeated": False},
            {"name": "SMOGATRON", "desc": "Señor del smog\nAhoga las ciudades en niebla tóxica", "unlocked": False, "defeated": False},
            {"name": "DEFORESTIX", "desc": "Talador de raíces\nDevora los pulmones del planeta", "unlocked": False, "defeated": False},
            {"name": "HEATCORE", "desc": "Maestro del calor\nConvierte ciudades en hornos", "unlocked": False, "defeated": False},
        ]

        # Buttons layout
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        if self.is_mobile:
            self.buttons = {
                "play": {"pos": (center_x, center_y - 80), "scale": (100, 100), "tex_hover": "Jugar"},
                "pvp": {"pos": (self.screen_width - 60, self.screen_height - 60), "scale": (50, 50), "tex_hover": "2 Jugadores"},
                "history": {"pos": (center_x - 80, center_y + 120), "scale": (30, 30), "tex_hover": "Historial"},
                "player": {"pos": (center_x + 80, center_y + 120), "scale": (30, 30), "tex_hover": "Jugador"},
                "settings": {"pos": (30, 30), "scale": (30, 30), "tex_hover": "Configuración"},
                "help": {"pos": (self.screen_width - 30, 30), "scale": (30, 30), "tex_hover": "Ayuda"},
            }
        else:
            self.buttons = {
                "play": {"pos": (center_x, center_y), "scale": (300, 300), "tex_hover": "Jugar"},
                "pvp": {"pos": (self.screen_width - 80, self.screen_height - 80), "scale": (70, 70), "tex_hover": "2 Jugadores"},
                "history": {"pos": (center_x - 220, center_y), "scale": (120, 120), "tex_hover": "Historial"},
                "player": {"pos": (center_x + 220, center_y), "scale": (120, 120), "tex_hover": "Jugador"},
                "settings": {"pos": (40, 40), "scale": (80, 80), "tex_hover": "Configuración"},
                "help": {"pos": (self.screen_width - 40, 40), "scale": (80, 80), "tex_hover": "Ayuda"},
            }

        # Panels state
        self.show_gaia_panel = False
        self.show_progress_panel = True

        # Popups
        self.help_popup = None
        self.settings_popup = None
        self.profile_popup = None

        # Profile manager (extracted component)
        self.profile_mgr = ProfileManager(self.screen, self.save_system, self.colors)

        # Load profiles and current data
        self._sync_profile_data()

        # Progress bars (extracted component)
        self._build_progress_bars()

        # Messages
        self.error_message = ""
        self.success_message = ""
        self.message_time = 0.0

        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    # Profile sync
    # ------------------------------------------------------------------

    def _sync_profile_data(self):
        """Sync game_data from the current profile."""
        if self.save_system.current_profile:
            p = self.save_system.current_profile
            self.game_data = {
                "player_points": p.get("total_points", 0),
                "planetary_progress": p.get("planetary_progress", self.game_data["planetary_progress"]),
                "levels_unlocked": p.get("levels", {}).get("unlocked", 1),
                "current_level": p.get("levels", {}).get("current", 1),
            }
            if "skin" in p:
                self.profile_mgr.skin_selector.selected_id = p["skin"]
            audio_manager.load_audio_settings_from_profile(self.save_system)
            # Update enemies
            for enemy in self.enemy_agents:
                enemy["defeated"] = enemy["name"] in p.get("enemies_defeated", [])
                idx = self.enemy_agents.index(enemy)
                enemy["unlocked"] = self.game_data["levels_unlocked"] >= idx + 1

    def _build_progress_bars(self):
        """Create ProgressBar widgets for the progress panel."""
        items = [
            ("Océanos limpiados:", "oceanos_limpiados", self.colors["ice_blue"]),
            ("Ozono restaurado:", "ozono_restaurado", self.colors["warning_orange"]),
            ("Aire purificado:", "aire_purificado", self.colors["hope_green"]),
            ("Bosques replantados:", "bosques_replantados", self.colors["hope_green"]),
            ("Ciudades enfriadas:", "ciudades_enfriadas", self.colors["ice_blue"]),
        ]
        self.progress_bars = []
        for label, key, color in items:
            pb = ProgressBar(0, 0, 0, 8,
                             fill_color=color, label=label,
                             label_font=self.font_small,
                             label_color=self.colors["text_white"],
                             value_color=color)
            pb.value = self.game_data["planetary_progress"].get(key, 0)
            self.progress_bars.append((key, pb))

    # ------------------------------------------------------------------
    # Drawing — Main Screen
    # ------------------------------------------------------------------

    def _draw_background(self):
        self.env_effects.draw(
            self.screen, self.animation_time,
            self.background_image, self.background_opacity,
        )

    def draw_title(self):
        title_text = "HOCKEY IS MELTING DOWN"
        subtitle_text = "Desafía a EcoNull y salva la Tierra"
        title_y = 100 if not self.is_mobile else 70
        subtitle_y = title_y + 40
        glow = abs(math.sin(self.animation_time * 4))

        # Glow layers
        for off in range(3, 0, -1):
            alpha = int(50 * glow)
            gc = (100 + off * 50, 150 + off * 30, 255, alpha)
            gs = self.font_title.render(title_text, True, gc)
            self.screen.blit(gs, gs.get_rect(center=(self.screen_width // 2, title_y)))

        # Shadow
        sh = self.font_title.render(title_text, True, (0, 0, 0))
        self.screen.blit(sh, sh.get_rect(center=(self.screen_width // 2 + 2, title_y + 2)))
        ss = self.font_subtitle.render(subtitle_text, True, (0, 0, 0))
        self.screen.blit(ss, ss.get_rect(center=(self.screen_width // 2 + 2, subtitle_y + 2)))

        # Main
        tc = (min(255, int(173 + 80 * glow)),
              min(255, int(216 + 40 * glow)),
              min(255, int(230 + 25 * glow)))
        ts = self.font_title.render(title_text, True, tc)
        self.screen.blit(ts, ts.get_rect(center=(self.screen_width // 2, title_y)))
        sts = self.font_subtitle.render(subtitle_text, True, self.colors["critical_red"])
        self.screen.blit(sts, sts.get_rect(center=(self.screen_width // 2, subtitle_y)))

        # Planetary status
        prefix = "ESTADO PLANETARIO: "
        critical = "CRÍTICO"
        ps = self.font_text.render(prefix, True, self.colors["critical_red"])
        pr = ps.get_rect(center=(self.screen_width // 2 - 40, title_y + 90))
        fire_g = abs(math.sin(self.animation_time * 4))
        cc = (255, int(100 + 155 * fire_g), 0)
        cs = self.font_text.render(critical, True, cc)
        cr = cs.get_rect(midleft=(pr.right, pr.centery))
        self.screen.blit(ps, pr)
        self.screen.blit(cs, cr)

    def draw_circular_button(self, button_obj: GameButton):
        button_obj.draw(self.screen)
        return button_obj.is_hovered()

    def draw_gaia_panel(self):
        if not self.show_gaia_panel:
            return
        pw = 300 if not self.is_mobile else self.screen_width - 40
        ph = 300 if not self.is_mobile else 200
        px = 20 if not self.is_mobile else 20
        py = self.screen_height // 2 - 50 if not self.is_mobile else self.screen_height // 2 + 50

        panel = Panel(px, py, pw, ph,
                      bg_color=self.colors["panel_dark"],
                      border_color=self.colors["hope_green"],
                      title="ARCHIVO GAIA",
                      title_font=self.font_subtitle,
                      title_color=self.colors["hope_green"])
        panel.draw(self.screen)

        cr = panel.content_rect
        gaia_lines = [
            "GAIA CORE - SISTEMA DE RESTAURACIÓN",
            "",
            "La Tierra agoniza, los polos se derriten,",
            "los bosques desaparecen y las ciudades",
            "arden bajo olas de calor implacables.",
            "",
            "Cinco agentes corruptos de EcoNull",
            "han tomado control del clima:",
            "",
            "🌊 SLICKWAVE - Emperador del plástico",
            "☀️ UVBLADE - Destructor del ozono",
            "🌪️ SMOGATRON - Señor del smog",
            "🌱 DEFORESTIX - Perdición de los bosques",
            "🔥 HEATCORE - Calor abominable",
        ]
        y_off = cr.y
        for line in gaia_lines:
            if line.strip():
                col = self.colors["text_gold"] if line.startswith(("🌊", "☀️", "🌪️", "🌱", "🔥")) else self.colors["text_white"]
                s = self.font_small.render(line, True, col)
                if y_off + s.get_height() > cr.bottom:
                    break
                self.screen.blit(s, (cr.x, y_off))
            y_off += 18

    def draw_progress_panel(self):
        if not self.show_progress_panel:
            return
        pw = 250 if not self.is_mobile else self.screen_width - 40
        ph = 210 if not self.is_mobile else 150
        px = self.screen_width - pw - 20 if not self.is_mobile else 20
        py = self.screen_height // 2 - 50 if not self.is_mobile else 20

        panel = Panel(px, py, pw, ph,
                      bg_color=self.colors["panel_dark"],
                      border_color=self.colors["ice_blue"],
                      title="PROGRESO PLANETARIO",
                      title_font=self.font_subtitle,
                      title_color=self.colors["ice_blue"])
        panel.draw(self.screen)

        cr = panel.content_rect
        bar_w = cr.width - 40
        y = cr.y
        for key, pb in self.progress_bars:
            pb.value = self.game_data["planetary_progress"].get(key, 0)
            pb.rect.x = cr.x
            pb.rect.y = y
            pb.rect.width = bar_w
            consumed = pb.draw(self.screen)
            y += consumed + 6

        # Points
        pts = f"Puntos Gaia: {self.game_data['player_points']}"
        psurf = self.font_text.render(pts, True, self.colors["text_gold"])
        self.screen.blit(psurf, (px + pw - psurf.get_width() - 10, py + ph - 25))

    def draw_climate_warning(self):
        warnings = [
            "Los glaciares pierden 280 mil millones de toneladas anuales",
            "La temperatura global ha aumentado 1.5°C desde 1880",
            "El nivel del mar sube 3.3mm cada año",
            "Quedan menos de 10 años para actuar",
        ]
        idx = int(self.animation_time / 3) % len(warnings)
        wf = font_cache.get(None, 22)
        ws = wf.render(warnings[idx], True, self.colors["text_white"])
        wr = ws.get_rect(center=(self.screen_width // 2, self.screen_height - 50))
        bg = pygame.Rect(wr.x - 15, wr.y - 10, wr.width + 30, wr.height + 20)
        pygame.draw.rect(self.screen, self.colors["warning_orange"], bg)
        pygame.draw.rect(self.screen, self.colors["text_white"], bg, 3)
        self.screen.blit(ws, wr)

    def draw_active_profile(self):
        if self.save_system.current_profile:
            name = self.save_system.current_profile["player_name"]
            text = f"Agente: {name}"
            surf = self.font_text.render(text, True, self.colors["text_gold"])
            rect = surf.get_rect(bottomleft=(20, self.screen_height - 20))
            bg = rect.inflate(20, 10)
            bgs = pygame.Surface(bg.size, pygame.SRCALPHA)
            bgs.fill((20, 20, 40, 200))
            self.screen.blit(bgs, bg)
            pygame.draw.rect(self.screen, self.colors["ice_blue"], bg, 1)
            self.screen.blit(surf, rect)

    def draw_messages(self):
        now = time.time()
        if self.error_message and (now - self.message_time) < 3:
            self._blit_msg(self.error_message, self.colors["critical_red"], (40, 0, 0))
        if self.success_message and (now - self.message_time) < 3:
            self._blit_msg(self.success_message, self.colors["hope_green"], (0, 40, 0))

    def _blit_msg(self, text, color, bg_tint):
        s = self.font_text.render(text, True, color)
        r = s.get_rect(center=(self.screen_width // 2, self.screen_height - 100))
        bg = pygame.Rect(r.x - 10, r.y - 5, r.width + 20, r.height + 10)
        pygame.draw.rect(self.screen, bg_tint, bg, border_radius=5)
        pygame.draw.rect(self.screen, color, bg, 2, border_radius=5)
        self.screen.blit(s, r)

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------

    def _show_profile_popup(self):
        """Show a popup asking the user to create/select a profile before playing."""
        self.profile_popup = PopUp(
            self.screen,
            title="⚠️ Perfil requerido",
            content=[
                "Necesitas un perfil para jugar.",
                "",
                "Crea o selecciona un perfil desde",
                "la sección de Jugador.",
            ],
            buttons=[
                {"text": "Ir a Perfiles", "action": "go_profiles"},
                {"text": "Cancelar", "action": "close"},
            ],
            popup_type="warning",
        )
        self.profile_popup.show()

    def handle_click(self, key):
        if key == "play":
            if not self.save_system.current_profile:
                self._show_profile_popup()
                return None
            return "level_select"
        elif key == "pvp":
            return "pvp_setup"
        elif key == "history":
            self.show_gaia_panel = not self.show_gaia_panel
        elif key == "player":
            self.current_screen = "profiles"
            return None
        elif key == "settings":
            self.settings_popup = create_settings_popup(self.screen, audio_manager)
            self.settings_popup.show()
            return None
        elif key == "help":
            self.help_popup = create_help_popup(self.screen, "home")
            self.help_popup.show()
            return "help"
        return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        running = True
        result = None

        # Build GameButton objects (image-based, replaces old Button)
        btn_objects: dict[str, GameButton] = {}
        for key, cfg in self.buttons.items():
            btn_objects[key] = image_button(
                asset_name=key,
                scale=cfg["scale"],
                position=cfg["pos"],
                hover_text=cfg["tex_hover"],
                name=key,
            )

        while running:
            dt = self.clock.tick(60) / 1000.0
            self.animation_time += dt
            self.env_effects.update(dt)

            # Update popups
            if self.help_popup:
                pr = self.help_popup.update(dt)
                if pr == "closed":
                    self.help_popup = None
            if self.settings_popup:
                pr = self.settings_popup.update(dt)
                if pr == "closed":
                    self.settings_popup = None
            if self.profile_popup:
                pr = self.profile_popup.update(dt)
                if pr == "closed":
                    self.profile_popup = None

            for event in pygame.event.get():
                audio_manager.process_event(event)

                if event.type == pygame.QUIT:
                    running = False
                    result = "exit"
                    continue

                # Popup events
                if self.profile_popup and self.profile_popup.is_visible():
                    pa = self.profile_popup.handle_event(event)
                    if pa == "go_profiles":
                        self.profile_popup.close()
                        self.profile_popup = None
                        self.current_screen = "profiles"
                    continue
                if self.help_popup and self.help_popup.is_visible():
                    self.help_popup.handle_event(event)
                    continue
                if self.settings_popup and self.settings_popup.is_visible():
                    pa = self.settings_popup.handle_event(event)
                    if pa:
                        self._handle_settings_action(pa)
                    continue

                # Screen-specific events
                if self.current_screen == "main":
                    # Dispatch event to all buttons
                    for key, bo in btn_objects.items():
                        if bo.update(event, dt):
                            action = self.handle_click(key)
                            if action in ("level_select", "pvp_setup"):
                                if self.save_system.current_profile:
                                    self.save_system.save_current_profile()
                                result = action
                                running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        action = self.handle_click("play")
                        if action == "level_select":
                            result = action
                            running = False

                elif self.current_screen == "profiles":
                    self._handle_profile_events(event)

            # Draw
            # Advance hover animations once per frame
            for bo in btn_objects.values():
                bo.update_animation(dt)

            if self.current_screen == "main":
                self._draw_background()
                self.draw_title()
                for key, bo in btn_objects.items():
                    self.draw_circular_button(bo)
                self.draw_gaia_panel()
                self.draw_progress_panel()
                self.draw_climate_warning()
                self.draw_active_profile()
                if self.help_popup:
                    self.help_popup.draw()
                if self.profile_popup:
                    self.profile_popup.draw()

            elif self.current_screen == "profiles":
                self._draw_profile_screen()

            self.draw_messages()
            if self.settings_popup:
                self.settings_popup.draw()

            pygame.display.flip()

        return result if result else "exit"

    # ------------------------------------------------------------------
    # Profile sub-screen delegation
    # ------------------------------------------------------------------

    def _draw_profile_screen(self):
        mgr = self.profile_mgr
        if mgr.current_screen == "list":
            mgr.draw_list(self._draw_background)
        elif mgr.current_screen == "create":
            mgr.draw_create(self._draw_background, self.animation_time)
        elif mgr.current_screen == "skin":
            mgr.draw_skin_screen(self._draw_background, self.animation_time)

    def _handle_profile_events(self, event):
        mgr = self.profile_mgr
        # Use cached UI rects from the last draw pass — avoids double-
        # rendering the entire profile screen just to obtain hit rects.
        ui = mgr.cached_ui
        if ui is None:
            return
        if mgr.current_screen == "list":
            action = mgr.handle_list_events(event, ui)
            if action == "back":
                self.current_screen = "main"
                self._sync_profile_data()
        elif mgr.current_screen == "create":
            action = mgr.handle_create_events(event, ui)
            if action == "back":
                self.current_screen = "main"
                self._sync_profile_data()
        elif mgr.current_screen == "skin":
            mgr.handle_skin_events(event, ui)

    # ------------------------------------------------------------------
    # Settings popup actions
    # ------------------------------------------------------------------

    def _handle_settings_action(self, action):
        if action == "toggle_music":
            audio_manager.toggle_music()
            audio_manager.save_audio_settings_to_profile(self.save_system)
            self.settings_popup = create_settings_popup(self.screen, audio_manager)
            self.settings_popup.show()
        elif action == "toggle_sfx":
            audio_manager.toggle_sfx()
            audio_manager.save_audio_settings_to_profile(self.save_system)
            self.settings_popup = create_settings_popup(self.screen, audio_manager)
            self.settings_popup.show()
        elif action == "save_settings":
            mv, sv = 0.5, 0.5
            for el in self.settings_popup.interactive_elements:
                if el["id"] == "music_volume":
                    mv = el["value"]
                elif el["id"] == "sfx_volume":
                    sv = el["value"]
            audio_manager.set_music_volume(mv)
            audio_manager.set_sfx_volume(sv)
            audio_manager.save_audio_settings_to_profile(self.save_system)
            self.settings_popup.close()
        elif action == "cancel":
            audio_manager.load_audio_settings_from_profile(self.save_system)
            self.settings_popup.close()
        elif isinstance(action, dict) and action.get("action") == "slider_change":
            sid = action.get("id")
            val = action.get("value")
            if sid == "music_volume":
                audio_manager.set_music_volume(val)
            elif sid == "sfx_volume":
                audio_manager.set_sfx_volume(val)
                if int(time.time() * 2) % 2 == 0:
                    audio_manager.play_sound_effect("button_hover")

    # ------------------------------------------------------------------
    # Backward-compat helpers
    # ------------------------------------------------------------------

    @property
    def selected_skin_id(self):
        return self.profile_mgr.skin_selector.selected_id

    @selected_skin_id.setter
    def selected_skin_id(self, v):
        self.profile_mgr.skin_selector.selected_id = v

    @property
    def available_skins(self):
        return self.profile_mgr.skin_selector.skins

    def load_profiles(self):
        self.profile_mgr.load_profiles()

    def load_selected_profile(self):
        return self.profile_mgr.load_selected_profile()

    def save_game_progress(self):
        if not self.save_system.current_profile:
            self.error_message = "No hay un perfil activo para guardar."
            self.message_time = time.time()
            return False
        game_data = {
            "points": self.game_data["player_points"],
            "planetary_progress": self.game_data["planetary_progress"],
            "level_completed": self.game_data["current_level"] - 1,
            "stats": {"games_played": 1, "time_played": 60},
        }
        for enemy in self.enemy_agents:
            if enemy["defeated"]:
                game_data["enemy_defeated"] = enemy["name"]
                break
        if self.save_system.update_game_progress(game_data):
            self.success_message = "¡Progreso guardado!"
            self.message_time = time.time()
            return True
        else:
            self.error_message = "Error al guardar el progreso."
            self.message_time = time.time()
            return False
