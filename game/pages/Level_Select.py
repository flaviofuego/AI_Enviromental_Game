"""
Level Selection screen for Hockey Is Melting Down.
Refactored: uses Panel, TextRenderer components for cleaner code
and proper text-overflow handling in the level details modal.
"""
import pygame
import math
import random

from ..config.save_system import GameSaveSystem
from ..components.GameButton import GameButton, image_button, text_button
from ..components.LevelThumbnail import LevelThumbnail
from ..components.Card import Card
from ..components.AudioManager import audio_manager
from ..components.PopUp import PopUp
from ..components.modals import create_help_popup
from ..components.Panel import Panel
from ..components.TextRenderer import TextRenderer
from ..components.FontCache import font_cache


# ── Level definitions ────────────────────────────────────────────────
LEVEL_DATA = [
    {
        "id": 1,
        "name": "Basura en el Ártico",
        "enemy": "SLICKWAVE",
        "theme": "Plástico",
        "description": (
            "Los mares del norte están inundados de plástico. "
            "Enfréntate a SlickWave, el emperador del plástico."
        ),
        "challenge": "Recicla toneladas de desechos anotando goles",
        "icon": "🌊",
        "color_key": "ice_blue",
    },
    {
        "id": 2,
        "name": "Agujero de Ozono",
        "enemy": "UVBLADE",
        "theme": "Gases CFC",
        "description": (
            "Los gases CFC han abierto un cráter en el cielo antártico. "
            "UVBlade controla la radiación."
        ),
        "challenge": "Restaura el escudo protector anotando en las porterías moleculares",
        "icon": "☀️",
        "color_key": "warning_orange",
    },
    {
        "id": 3,
        "name": "Tormenta de Smog",
        "enemy": "SMOGATRON",
        "theme": "Aire Contaminado",
        "description": (
            "La niebla tóxica asfixia las ciudades. Smogatron, el emperador "
            "del smog, bloquea el aire limpio."
        ),
        "challenge": "Activa los filtros de aire con cada gol para purificar la atmósfera",
        "icon": "☁️",
        "color_key": "smog_gray",
    },
    {
        "id": 4,
        "name": "Bosque Desvanecido",
        "enemy": "DEFORESTIX",
        "theme": "Deforestación",
        "description": (
            "Los pulmones del planeta desaparecen rápidamente. "
            "Deforestix arrasa con todo a su paso."
        ),
        "challenge": "Planta miles de árboles virtuales con cada victoria",
        "icon": "🌳",
        "color_key": "hope_green",
    },
    {
        "id": 5,
        "name": "Isla de Calor Urbano",
        "enemy": "HEATCORE",
        "theme": "Calentamiento Urbano",
        "description": (
            "Las ciudades son hornos de asfalto. HeatCore eleva "
            "las temperaturas a niveles insoportables."
        ),
        "challenge": "Enfría las ciudades con disparos certeros",
        "icon": "🔥",
        "color_key": "critical_red",
    },
]


class LevelSelectScreen:
    """Level-selection screen with card grid, detail panel, and ambient particles."""

    # ── Colour palette ───────────────────────────────────────────────
    COLORS = {
        "bg_gradient_top": (173, 216, 230),
        "bg_gradient_bottom": (120, 50, 50),
        "ice_blue": (173, 216, 230),
        "critical_red": (220, 50, 50),
        "hope_green": (34, 139, 34),
        "warning_orange": (255, 140, 0),
        "text_white": (255, 255, 255),
        "text_gold": (255, 215, 0),
        "panel_dark": (20, 20, 40, 220),
        "button_active": (0, 100, 200),
        "button_hover": (0, 150, 255),
        "locked_gray": (140, 140, 140),
        "completed_green": (50, 180, 50),
        "level_card_bg": (40, 40, 60, 200),
        "smog_gray": (100, 100, 150),
    }

    # ── Init ─────────────────────────────────────────────────────────

    def __init__(self, save_system=None, screen=None):
        # Screen
        if screen is None:
            pygame.init()
            info = pygame.display.Info()
            self.screen_width = min(1200, info.current_w - 100)
            self.screen_height = min(800, info.current_h - 100)
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Hockey Is Melting Down - Selección de Niveles")
        else:
            self.screen = screen
            self.screen_width = screen.get_width()
            self.screen_height = screen.get_height()

        self.is_mobile = self.screen_height > self.screen_width

        # Background image
        try:
            self.background_image = pygame.image.load(
                "game/assets/niveles/background_levels.png"
            )
            self.background_image = pygame.transform.scale(
                self.background_image, (self.screen_width, self.screen_height)
            )
            self.background_opacity = 200

            # Pre-cache the alpha-applied background so we don't copy
            # and set_alpha every frame.
            self._bg_with_alpha = self.background_image.copy()
            self._bg_with_alpha.set_alpha(self.background_opacity)
        except (pygame.error, FileNotFoundError):
            self.background_image = None
            self._bg_with_alpha = None

        # Save system
        self.save_system = save_system if save_system else GameSaveSystem()
        self.current_profile = self.save_system.current_profile
        if not self.current_profile:
            print("Error: No hay perfil activo en el sistema de guardado")

        # Selection state
        self.selected_level_index = -1
        self.selected_level = None
        self.hover_level_index = -1
        self.show_level_info = False
        self.animation_time = 0.0

        # Fonts (via FontCache)
        m = self.is_mobile
        self.font_title = font_cache.get(None, 48 if not m else 36)
        self.font_subtitle = font_cache.get(None, 24 if not m else 20)
        self.font_text = font_cache.get(None, 24 if not m else 20)
        self.font_small = font_cache.get(None, 20 if not m else 18)

        # Text renderer — used in the detail panel to prevent overflow
        self.text_renderer = TextRenderer(self.font_small, self.COLORS["text_white"])

        # Build runtime levels (add mutable state & resolved colour)
        self.levels = []
        for ld in LEVEL_DATA:
            self.levels.append({
                **ld,
                "color": self.COLORS.get(ld["color_key"], (200, 200, 200)),
                "unlocked": ld["id"] == 1,
                "completed": False,
            })

        # Nav buttons (GameButton instances)
        btn_w, btn_h, margin = 150, 40, 20
        self.btn_back = text_button(
            text="Volver al Menú",
            position=(margin, self.screen_height - btn_h - margin),
            size=(btn_w, btn_h),
            bg_color=self.COLORS["ice_blue"],
            hover_color=self.COLORS["button_hover"],
            font_size=20 if not m else 18,
        )
        self.btn_play = text_button(
            text="¡Jugar Nivel!",
            position=(self.screen_width - btn_w - margin,
                      self.screen_height - btn_h - margin),
            size=(btn_w, btn_h),
            bg_color=self.COLORS["hope_green"],
            hover_color=(80, 200, 80),
            font_size=20 if not m else 18,
        )
        self.btn_help = image_button(
            asset_name="help",
            scale=(40, 40),
            position=(self.screen_width - 40, 40),
            hover_text="Ayuda",
            name="help",
        )
        self._nav_buttons = {"back": self.btn_back, "play": self.btn_play}

        # Particles
        self.particles = self._create_particles(30)

        # Cards & thumbnail widgets
        self.cards = {}
        self.thumbnails = {}
        for level in self.levels:
            lid = level["id"]
            self.cards[lid] = Card(
                max_width=400 if not self.is_mobile else self.screen_width - 40,
                spacing=10,
                padding=10,
            )
            self.thumbnails[lid] = LevelThumbnail(lid)

        # Messages
        self.message = ""
        self.message_time = 0
        self.message_type = "info"

        # Help popup
        self.help_popup = None

        # Clock
        self.clock = pygame.time.Clock()

        # Sync level status from profile
        self.load_levels_status()

        # Pre-load audio
        audio_manager.preload_audio_for_screen("level_select")

    # ── Level status sync ────────────────────────────────────────────

    def load_levels_status(self):
        """Update level unlock/completed state from the current profile."""
        if self.save_system and self.current_profile:
            try:
                updated = self.save_system.load_profile(
                    self.current_profile["profile_id"]
                )
                if updated:
                    self.current_profile = updated
            except Exception as e:
                print(f"Error recargando perfil: {e}")

        if self.current_profile:
            unlocked = self.current_profile.get("levels", {}).get("unlocked", 1)
            completed = self.current_profile.get("levels", {}).get("completed", [])
            for level in self.levels:
                level["unlocked"] = level["id"] <= unlocked
                level["completed"] = level["id"] in completed

    # ── Particles ────────────────────────────────────────────────────

    def _create_particles(self, count: int) -> list[dict]:
        particles = []
        for _ in range(count):
            particles.append({
                "x": random.randint(0, self.screen_width),
                "y": random.randint(0, self.screen_height),
                "size": random.randint(1, 3),
                "speed": random.uniform(0.5, 1.5),
                "angle": random.uniform(0, 2 * math.pi),
                "color": random.choice([
                    self.COLORS["ice_blue"],
                    self.COLORS["hope_green"],
                    self.COLORS["warning_orange"],
                    self.COLORS["smog_gray"],
                ]),
            })
        return particles

    def _update_particles(self):
        for p in self.particles:
            p["x"] += math.cos(p["angle"]) * p["speed"]
            p["y"] += math.sin(p["angle"]) * p["speed"]
            # Wrap around
            if p["x"] < 0:
                p["x"] = self.screen_width
            elif p["x"] > self.screen_width:
                p["x"] = 0
            if p["y"] < 0:
                p["y"] = self.screen_height
            elif p["y"] > self.screen_height:
                p["y"] = 0

    # ── Drawing helpers ──────────────────────────────────────────────

    def _draw_background(self):
        if self._bg_with_alpha:
            self.screen.blit(self._bg_with_alpha, (0, 0))
        else:
            for y in range(self.screen_height):
                ratio = y / self.screen_height
                c = [
                    int(self.COLORS["bg_gradient_top"][i] * (1 - ratio)
                        + self.COLORS["bg_gradient_bottom"][i] * ratio)
                    for i in range(3)
                ]
                pygame.draw.line(self.screen, c, (0, y), (self.screen_width, y))

        # Particles
        for p in self.particles:
            alpha = int(128 + 127 * math.sin(self.animation_time * 2 + p["x"] * 0.01))
            pygame.draw.circle(
                self.screen,
                (*p["color"], alpha),
                (int(p["x"]), int(p["y"])),
                p["size"],
            )

    def _draw_title(self):
        title_text = "SELECCIONA TU MISIÓN"
        title_y = 50 if not self.is_mobile else 30
        glow = abs(math.sin(self.animation_time * 3))

        # Shadow
        shadow = self.font_title.render(title_text, True, (0, 0, 0))
        self.screen.blit(shadow, shadow.get_rect(center=(self.screen_width // 2 + 2, title_y + 2)))

        # Glowing title
        tc = (
            min(255, int(173 + 80 * glow)),
            min(255, int(216 + 40 * glow)),
            min(255, int(230 + 25 * glow)),
        )
        ts = self.font_title.render(title_text, True, tc)
        self.screen.blit(ts, ts.get_rect(center=(self.screen_width // 2, title_y)))

        # Subtitle
        sub = "Restaura el equilibrio climático, misión por misión"
        ss = self.font_subtitle.render(sub, True, self.COLORS["text_gold"])
        self.screen.blit(ss, ss.get_rect(center=(self.screen_width // 2, title_y + 40)))

        # Profile label
        if self.current_profile:
            ps = self.font_text.render(
                f"Agente: {self.current_profile['player_name']}",
                True,
                self.COLORS["text_white"],
            )
            self.screen.blit(ps, (20, 20))

    # ── Level cards ──────────────────────────────────────────────────

    def _draw_level_cards(self):
        cards_per_row = 1 if self.is_mobile else 3
        gap = 20
        start_y = 150
        num_rows = math.ceil(len(self.levels) / cards_per_row)

        for i, level in enumerate(self.levels):
            # Status label
            if not level["unlocked"]:
                status_text, status_color = "BLOQUEADO", self.COLORS["locked_gray"]
            elif level["completed"]:
                status_text, status_color = "COMPLETADO", self.COLORS["completed_green"]
            else:
                status_text, status_color = "DISPONIBLE", self.COLORS["ice_blue"]

            card = self.cards[level["id"]]

            if self.is_mobile:
                pos = (20, start_y + (card.height + gap) * i)
            else:
                row = i // cards_per_row
                col = i % cards_per_row
                if row == num_rows - 1:
                    cards_last = len(self.levels) - row * cards_per_row
                else:
                    cards_last = cards_per_row
                total_w = (card.width + gap) * cards_last - gap
                sx = (self.screen_width - total_w) // 2
                pos = (sx + (card.width + gap) * col, start_y + (card.height + gap) * row)

            card_rect = card.draw(
                self.screen,
                pos,
                self.thumbnails[level["id"]].image,
                status_text,
                status_color,
                i == self.selected_level_index,
                abs(math.sin(self.animation_time * 10))
                if i == self.selected_level_index
                else 0,
            )
            level["rect"] = card_rect

    # ── Level detail panel (text-overflow fix) ───────────────────────

    def _draw_level_details(self):
        """Draw the detail panel for the selected level.

        Uses TextRenderer to word-wrap and clip all text sections so
        they never overflow the panel boundaries.
        """
        if self.selected_level_index < 0 or not self.show_level_info:
            return

        level = self.levels[self.selected_level_index]

        # Panel geometry
        pw = 500 if not self.is_mobile else self.screen_width - 40
        ph = 250
        px = (self.screen_width - pw) // 2
        py = self.screen_height - ph - 80

        # Draw panel using Panel component
        panel = Panel(
            px, py, pw, ph,
            bg_color=self.COLORS["panel_dark"],
            border_color=level["color"],
            border_width=2,
            title=f"Nivel {level['id']}: {level['name']}",
            title_font=self.font_subtitle,
            title_color=level["color"],
        )
        panel.draw(self.screen)

        cr = panel.content_rect
        content_w = cr.width - 10  # small inset
        y_cursor = cr.y

        # Enemy info line
        enemy_text = f"Enemigo: {level['enemy']} - {level['theme']}"
        enemy_surf = self.font_text.render(enemy_text, True, self.COLORS["text_gold"])
        if y_cursor + enemy_surf.get_height() <= cr.bottom:
            self.screen.blit(enemy_surf, (cr.x, y_cursor))
        y_cursor += enemy_surf.get_height() + 8

        # Remaining vertical space for wrapped text sections
        remaining = cr.bottom - y_cursor - 35  # reserve 35px for status line
        if remaining < 20:
            remaining = 20

        # Allocate space: description gets 60%, challenge gets 40%
        desc_budget = int(remaining * 0.6)
        challenge_budget = remaining - desc_budget

        # Description (word-wrapped, clipped)
        consumed = self.text_renderer.render_text(
            self.screen,
            level["description"],
            cr.x, y_cursor,
            content_w,
            max_height=desc_budget,
            color=self.COLORS["text_white"],
        )
        y_cursor += consumed + 6

        # Challenge (word-wrapped, clipped)
        consumed = self.text_renderer.render_text(
            self.screen,
            "Desafío: " + level["challenge"],
            cr.x, y_cursor,
            content_w,
            max_height=challenge_budget,
            color=self.COLORS["hope_green"],
        )

        # Status line — always at bottom of panel
        if not level["unlocked"]:
            st_text = "¡Completa el nivel anterior para desbloquear este!"
            st_color = self.COLORS["locked_gray"]
        elif level["completed"]:
            st_text = "¡Nivel completado! Puedes volver a jugarlo."
            st_color = self.COLORS["completed_green"]
        else:
            st_text = "¡Nivel listo para jugar! Ayuda a restaurar el planeta."
            st_color = self.COLORS["ice_blue"]

        # Wrap status text too so it doesn't overflow horizontally
        status_lines = self.text_renderer.wrap_text(st_text, content_w)
        status_y = py + ph - 15 - len(status_lines) * (self.font_small.get_linesize() + 4)
        self.text_renderer.render_lines(
            self.screen, status_lines,
            cr.x, max(status_y, y_cursor + 10),
            max_height=30,
            color=st_color,
        )

    # ── Nav buttons ──────────────────────────────────────────────────

    def _draw_buttons(self):
        for btn in self._nav_buttons.values():
            btn.draw(self.screen)

        # Disable overlay for play when no valid level
        if self.selected_level_index < 0 or not self.levels[self.selected_level_index]["unlocked"]:
            overlay = pygame.Surface(self.btn_play.base_rect.size, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, self.btn_play.base_rect)

    # ── Messages ─────────────────────────────────────────────────────

    def show_message(self, message, message_type="info"):
        self.message = message
        self.message_type = message_type
        self.message_time = pygame.time.get_ticks()

    def _draw_message(self):
        if not self.message or self.message_time == 0:
            return
        if pygame.time.get_ticks() - self.message_time > 3000:
            self.message = ""
            self.message_time = 0
            return

        type_colors = {
            "error": (220, 50, 50, 200),
            "success": (50, 180, 50, 200),
            "info": (50, 150, 200, 200),
        }
        bg_col = type_colors.get(self.message_type, type_colors["info"])
        text_col = (255, 255, 255)

        ms = self.font_text.render(self.message, True, text_col)
        pad = 20
        mw, mh = ms.get_width() + pad * 2, ms.get_height() + pad * 2
        mx = (self.screen_width - mw) // 2
        my = 100

        panel = pygame.Surface((mw, mh), pygame.SRCALPHA)
        panel.fill(bg_col)
        pygame.draw.rect(panel, text_col, (0, 0, mw, mh), 2)
        panel.blit(ms, (pad, pad))
        self.screen.blit(panel, (mx, my))

    # ── Event handling ───────────────────────────────────────────────

    def _handle_click(self, pos):
        # Level card clicks
        for i, level in enumerate(self.levels):
            if "rect" in level and level["rect"].collidepoint(pos):
                if level["unlocked"]:
                    audio_manager.play_sound_effect("button_click")
                    self.selected_level_index = i
                    self.selected_level = level
                    self.show_level_info = True
                    return True
                else:
                    audio_manager.play_sound_effect("button_click", volume_override=0.2)
                    self.show_message(
                        "Este nivel está bloqueado. ¡Completa los niveles anteriores!",
                        "error",
                    )
                    return False

        # Click on empty space — dismiss detail
        self.show_level_info = False
        return False

    def _handle_hover(self, pos):
        old = self.hover_level_index
        self.hover_level_index = -1
        for i, level in enumerate(self.levels):
            if "rect" in level and level["rect"].collidepoint(pos):
                self.hover_level_index = i
                if old != i and level["unlocked"]:
                    audio_manager.play_sound_effect("button_hover", volume_override=0.2)
                return

    # ── Start level ──────────────────────────────────────────────────

    def start_level(self, level_id):
        """Launch a level via main_hub.start_game."""
        level_data = None
        for lv in self.levels:
            if lv["id"] == level_id:
                level_data = lv
                break

        if not level_data:
            self.show_message(f"Error: Nivel {level_id} no encontrado", "error")
            return None
        if not level_data["unlocked"]:
            self.show_message("Nivel bloqueado. Completa el nivel anterior.", "error")
            return None

        self.show_message(f"Iniciando {level_data['name']}...", "info")
        self._draw_background()
        self._draw_title()
        self._draw_level_cards()
        self._draw_buttons()
        self._draw_message()
        pygame.display.flip()
        pygame.time.wait(500)

        try:
            from game.main_hub import start_game

            result = start_game(
                use_rl=True,
                screen=self.screen,
                level_config={"level_id": level_id},
                save_system=self.save_system,
            )
            print(f"Resultado del juego: {result}")
            self.load_levels_status()

            if result and isinstance(result, dict):
                if result.get("victory", False):
                    self.show_message(
                        f"¡Felicidades! Has completado {level_data['name']}",
                        "success",
                    )
                else:
                    self.show_message(
                        f"Intenta de nuevo. {level_data['name']} te espera.",
                        "info",
                    )
            else:
                self.show_message("Juego cancelado", "info")

            return result

        except ImportError as e:
            self.show_message(f"Error: No se pudo cargar el juego: {e}", "error")
            return None
        except Exception as e:
            self.show_message(f"Error inesperado: {e}", "error")
            return None

    # ── Backward compat helper ───────────────────────────────────────

    def wrap_text(self, text, font, max_width):
        """Legacy wrap — delegates to TextRenderer."""
        tr = TextRenderer(font, self.COLORS["text_white"])
        return tr.wrap_text(text, max_width)

    # ── Main loop ────────────────────────────────────────────────────

    def run(self):
        running = True
        result = None

        while running:
            dt = self.clock.tick(60) / 1000.0
            self.animation_time += dt
            self._update_particles()

            # Popup update
            if self.help_popup:
                pr = self.help_popup.update(dt)
                if pr == "closed":
                    self.help_popup = None

            # Events
            for event in pygame.event.get():
                audio_manager.process_event(event)

                if event.type == pygame.QUIT:
                    return "exit"

                if self.help_popup and self.help_popup.is_visible():
                    self.help_popup.handle_event(event)
                    continue

                # Dispatch to GameButton nav buttons
                if self.btn_back.update(event, dt):
                    result = "back_to_menu"
                    running = False
                    continue
                if self.btn_play.update(event, dt):
                    if (
                        self.selected_level_index >= 0
                        and self.levels[self.selected_level_index]["unlocked"]
                    ):
                        lid = self.levels[self.selected_level_index]["id"]
                        self.start_level(lid)
                    else:
                        self.show_message(
                            "Selecciona un nivel disponible para jugar", "error")
                    continue
                if self.btn_help.update(event, dt):
                    self.help_popup = create_help_popup(self.screen, "level_select")
                    self.help_popup.show()
                    continue

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    action = self._handle_click(event.pos)
                    if isinstance(action, str):
                        result = action
                        running = False
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_hover(event.pos)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return "exit"

            # Advance hover animations
            self.btn_back.update_animation(dt)
            self.btn_play.update_animation(dt)
            self.btn_help.update_animation(dt)

            # Draw
            self._draw_background()
            self._draw_title()
            self._draw_level_cards()
            if self.show_level_info:
                self._draw_level_details()
            self._draw_buttons()
            self.btn_help.draw(self.screen)
            self._draw_message()
            if self.help_popup:
                self.help_popup.draw()

            pygame.display.flip()

        return result


# Standalone test
if __name__ == "__main__":
    screen = LevelSelectScreen()
    result = screen.run()
    print(f"Result: {result}")
