import pygame
import math

from .IconRenderer import IconRenderer
from .FontCache import font_cache

class PopUp:
    """
    Reusable overlay pop-up component.

    Responsibilities: rendering, animation, and generic event dispatching.
    Content-specific modal construction lives in ``modals.py``.
    """

    def __init__(self, screen, title="", content=None, buttons=None, popup_type="info"):
        """
        Args:
            screen: pygame surface to draw on.
            title: Header text.
            content: List of text lines (or a single string).
            buttons: List of ``{"text": ..., "action": ...}`` dicts.
            popup_type: One of ``"info"``, ``"warning"``, ``"error"``, ``"help"``.
        """
        self.screen = screen
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()

        self.title = title
        self.content = (content if isinstance(content, list)
                        else [content] if content else [])
        self.popup_type = popup_type
        self.visible = False
        self.closing = False

        self.buttons = buttons if buttons else [{"text": "OK", "action": "close"}]

        self._setup_visual_config()

        # Animation
        self.animation_time = 0
        self.fade_duration = 0.3
        self.scale_animation = True
        self.current_scale = 0.0
        self.target_scale = 1.0

        # Hover / click state
        self.hovered_button = -1
        self.button_rects: list[pygame.Rect] = []

        # Interactive elements (sliders, etc.)
        self.interactive_elements: list[dict] = []
        self.active_element = None

        # Fonts (cached via FontCache singleton)
        self.font_title = font_cache.get(None, 28)
        self.font_content = font_cache.get(None, 20)
        self.font_button = font_cache.get(None, 18)

        # Cached overlay (created once at correct size)
        self._overlay: pygame.Surface | None = None

        self.calculate_dimensions()

    def add_slider(self, x, y, width, height, min_value, max_value, current_value, id):
        """Add an interactive slider element."""
        slider = {
            "type": "slider",
            "rect": pygame.Rect(x, y, width, height),
            "min": min_value,
            "max": max_value,
            "value": current_value,
            "id": id,
            "dragging": False,
        }
        self.interactive_elements.append(slider)
        return slider

    def _setup_visual_config(self):
        """Set colour palette by popup type. Icons are drawn via IconRenderer."""
        configs = {
            "info": {
                "bg_color": (20, 30, 50, 240),
                "border_color": (100, 150, 200),
                "title_color": (173, 216, 230),
                "text_color": (255, 255, 255),
                "button_color": (60, 100, 140),
                "button_hover": (80, 120, 160),
            },
            "warning": {
                "bg_color": (50, 40, 20, 240),
                "border_color": (255, 200, 100),
                "title_color": (255, 215, 0),
                "text_color": (255, 255, 255),
                "button_color": (200, 140, 60),
                "button_hover": (220, 160, 80),
            },
            "error": {
                "bg_color": (50, 20, 20, 240),
                "border_color": (220, 100, 100),
                "title_color": (255, 100, 100),
                "text_color": (255, 255, 255),
                "button_color": (180, 60, 60),
                "button_hover": (200, 80, 80),
            },
            "help": {
                "bg_color": (20, 40, 30, 240),
                "border_color": (100, 200, 150),
                "title_color": (100, 255, 150),
                "text_color": (240, 255, 240),
                "button_color": (60, 140, 100),
                "button_hover": (80, 160, 120),
            },
        }

        config = configs.get(self.popup_type, configs["info"])
        for key, value in config.items():
            setattr(self, key, value)
    
    def calculate_dimensions(self):
        """Calcular dimensiones del pop-up basado en contenido"""
        # Dimensiones mínimas y máximas
        min_width = 300
        max_width = min(600, self.screen_width - 100)
        min_height = 150
        max_height = min(500, self.screen_height - 100)
        
        # Calcular ancho basado en contenido
        title_width = self.font_title.size(self.title)[0] if self.title else 0
        content_width = 0
        for line in self.content:
            line_width = self.font_content.size(line)[0]
            content_width = max(content_width, line_width)
        
        # Calcular ancho de botones
        button_width = 0
        for button in self.buttons:
            btn_width = self.font_button.size(button["text"])[0] + 40
            button_width += btn_width + 10
        
        # Determinar ancho final
        content_width = max(title_width, content_width, button_width)
        self.width = min(max(content_width + 60, min_width), max_width)
        
        # Calcular alto basado en contenido
        title_height = 40 if self.title else 0
        content_height = len(self.content) * 25 + 20
        button_height = 50
        padding = 40
        
        self.height = min(title_height + content_height + button_height + padding, max_height)
        
        # Posición centrada
        self.x = (self.screen_width - self.width) // 2
        self.y = (self.screen_height - self.height) // 2
        
        # Verificar si necesita scroll
        total_content_height = title_height + len(self.content) * 25 + button_height + padding
        self.needs_scroll = total_content_height > self.height
        self.scroll_offset = 0
        self.max_scroll = max(0, total_content_height - self.height)
    
    def show(self):
        """Mostrar el pop-up con animación"""
        self.visible = True
        self.closing = False
        self.animation_time = 0
        self.current_scale = 0.0
        
        # Reproducir sonido según tipo
        from .AudioManager import audio_manager
        if self.popup_type == "error":
            audio_manager.play_sound_effect("button_click", volume_override=0.3)
        else:
            audio_manager.play_sound_effect("button_hover", volume_override=0.2)
    
    def close(self):
        """Cerrar el pop-up con animación"""
        self.closing = True
        self.animation_time = 0
    
    def update(self, dt):
        """Actualizar animaciones del pop-up"""
        if not self.visible:
            return
        
        self.animation_time += dt
        
        if self.closing:
            # Animación de cierre
            progress = min(self.animation_time / self.fade_duration, 1.0)
            self.current_scale = self.target_scale * (1.0 - progress)
            
            if progress >= 1.0:
                self.visible = False
                self.closing = False
                return "closed"
        else:
            # Animación de apertura
            if self.scale_animation:
                progress = min(self.animation_time / self.fade_duration, 1.0)
                # Efecto de rebote suave
                if progress < 1.0:
                    bounce = 1.0 + 0.1 * math.sin(progress * math.pi * 2)
                    self.current_scale = progress * bounce
                else:
                    self.current_scale = self.target_scale
        
        return None
    
    def handle_event(self, event):
        """Process a single pygame event. Returns an action string/dict or None."""
        if not self.visible or self.closing:
            return None

        if event.type == pygame.MOUSEMOTION:
            # Update button hover state
            self._update_hover(event.pos)
            # Drag active sliders
            for element in self.interactive_elements:
                if element["type"] == "slider" and element["dragging"]:
                    self._update_slider_value(element, event.pos)
                    return {"action": "slider_change", "id": element["id"],
                            "value": element["value"]}
            return None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Check interactive elements first
                for element in self.interactive_elements:
                    element_rect = pygame.Rect(
                        self.x + element["rect"].x,
                        self.y + element["rect"].y,
                        element["rect"].width,
                        element["rect"].height,
                    )
                    if element_rect.collidepoint(event.pos):
                        if element["type"] == "slider":
                            element["dragging"] = True
                            self._update_slider_value(element, event.pos)
                            return {"action": "slider_change", "id": element["id"],
                                    "value": element["value"]}
                # Then buttons
                return self.handle_click(event.pos)
            if event.button == 4:
                self.scroll_up()
            elif event.button == 5:
                self.scroll_down()
            return None

        if event.type == pygame.MOUSEBUTTONUP:
            for element in self.interactive_elements:
                if element["type"] == "slider" and element["dragging"]:
                    element["dragging"] = False
                    return {"action": "slider_final", "id": element["id"],
                            "value": element["value"]}
            return None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.close()
                return "close"
            if event.key == pygame.K_RETURN and self.buttons:
                return self.buttons[0]["action"]
            if event.key == pygame.K_UP:
                self.scroll_up()
            elif event.key == pygame.K_DOWN:
                self.scroll_down()

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_hover(self, pos):
        """Update hovered_button index based on mouse position."""
        old = self.hovered_button
        self.hovered_button = -1
        for i, rect in enumerate(self.button_rects):
            if rect.collidepoint(pos):
                self.hovered_button = i
                break
        if old != self.hovered_button and self.hovered_button != -1:
            from .AudioManager import audio_manager
            audio_manager.play_sound_effect("button_hover", volume_override=0.1)

    def _update_slider_value(self, element, pos):
        """Recalculate slider value from mouse position."""
        element_rect = pygame.Rect(
            self.x + element["rect"].x,
            self.y + element["rect"].y,
            element["rect"].width,
            element["rect"].height,
        )
        rel_x = max(0, min(pos[0] - element_rect.x, element_rect.width))
        element["value"] = (element["min"]
                            + (element["max"] - element["min"])
                            * (rel_x / element["rect"].width))
    
    def handle_click(self, pos):
        """Handle a left-click at *pos*. Returns action string or None."""
        for i, rect in enumerate(self.button_rects):
            if rect.collidepoint(pos):
                from .AudioManager import audio_manager
                audio_manager.play_sound_effect("button_click", volume_override=0.3)

                action = self.buttons[i].get("action", "close")
                if action == "close":
                    self.close()
                return action

        # Click outside popup dismisses it
        popup_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        if not popup_rect.collidepoint(pos):
            self.close()
            return "close"

        return None
    
    def scroll_up(self):
        """Scroll hacia arriba"""
        if self.needs_scroll:
            self.scroll_offset = max(0, self.scroll_offset - 20)
    
    def scroll_down(self):
        """Scroll hacia abajo"""
        if self.needs_scroll:
            self.scroll_offset = min(self.max_scroll, self.scroll_offset + 20)
    
    def draw(self):
        """Draw the pop-up onto the screen."""
        if not self.visible:
            return

        # Cached dark overlay
        if self._overlay is None:
            self._overlay = pygame.Surface(
                (self.screen_width, self.screen_height), pygame.SRCALPHA
            )
            self._overlay.fill((0, 0, 0, 120))
        self.screen.blit(self._overlay, (0, 0))

        # Scaled dimensions for open/close animation
        scaled_width = int(self.width * self.current_scale)
        scaled_height = int(self.height * self.current_scale)
        scaled_x = self.x + (self.width - scaled_width) // 2
        scaled_y = self.y + (self.height - scaled_height) // 2

        if scaled_width <= 0 or scaled_height <= 0:
            return

        popup_surface = pygame.Surface(
            (scaled_width, scaled_height), pygame.SRCALPHA
        )
        popup_surface.fill(self.bg_color)

        # Glowing border
        glow_intensity = abs(math.sin(self.animation_time * 3)) * 0.3 + 0.7
        glow_color = tuple(
            min(255, int(c * glow_intensity)) for c in self.border_color[:3]
        )
        pygame.draw.rect(
            popup_surface, glow_color,
            (0, 0, scaled_width, scaled_height), 2,
        )

        # Use the base fonts during full-scale; use font_cache for
        # animation sizes to avoid creating new Font objects each frame.
        if self.current_scale < 1.0:
            sf = max(0.1, self.current_scale)
            scaled_font_title = font_cache.get(None, max(12, int(28 * sf)))
            scaled_font_content = font_cache.get(None, max(10, int(20 * sf)))
            scaled_font_button = font_cache.get(None, max(8, int(18 * sf)))
        else:
            scaled_font_title = self.font_title
            scaled_font_content = self.font_content
            scaled_font_button = self.font_button

        if self.current_scale > 0.3:
            self._draw_sliders(popup_surface, scaled_height)

            y_offset = 20 - self.scroll_offset

            # Title row: icon + text
            if self.title:
                icon_size = scaled_font_title.get_linesize()
                icon_surf = IconRenderer.get_popup_icon(self.popup_type, icon_size)
                title_surf = scaled_font_title.render(self.title, True, self.title_color)
                total_w = icon_size + 6 + title_surf.get_width()
                start_x = (scaled_width - total_w) // 2
                if -title_surf.get_height() < y_offset < scaled_height:
                    popup_surface.blit(icon_surf, (start_x, max(0, y_offset)))
                    popup_surface.blit(title_surf, (start_x + icon_size + 6, max(0, y_offset)))
                y_offset += 40

            # Separator
            if self.title and 0 < y_offset < scaled_height:
                pygame.draw.line(
                    popup_surface, self.border_color,
                    (20, min(y_offset, scaled_height - 5)),
                    (scaled_width - 20, min(y_offset, scaled_height - 5)),
                )
            y_offset += 10

            # Content lines
            for line in self.content:
                if y_offset > scaled_height:
                    break
                if y_offset > -25:
                    cs = scaled_font_content.render(line, True, self.text_color)
                    popup_surface.blit(cs, (20, max(0, y_offset)))
                y_offset += 25

            # Buttons
            self._draw_buttons(popup_surface, scaled_font_button,
                               scaled_width, scaled_height,
                               scaled_x, scaled_y, y_offset)

            # Scroll indicators (drawn arrows, not emoji)
            if self.needs_scroll and self.current_scale > 0.8:
                arrow_rect = pygame.Rect(scaled_width - 25, 2, 16, 16)
                if self.scroll_offset > 0:
                    IconRenderer.draw_arrow_up(
                        popup_surface, arrow_rect, self.text_color, padding=3)
                arrow_rect.y = scaled_height - 18
                if self.scroll_offset < self.max_scroll:
                    IconRenderer.draw_arrow_down(
                        popup_surface, arrow_rect, self.text_color, padding=3)

        self.screen.blit(popup_surface, (scaled_x, scaled_y))

    # ------------------------------------------------------------------
    # Draw sub-sections (extracted for readability)
    # ------------------------------------------------------------------

    def _draw_sliders(self, popup_surface, scaled_height):
        """Render interactive slider elements onto the popup surface."""
        for element in self.interactive_elements:
            if element["type"] != "slider":
                continue
            slider_rect = pygame.Rect(
                element["rect"].x,
                max(0, element["rect"].y - self.scroll_offset),
                element["rect"].width,
                element["rect"].height,
            )
            if slider_rect.y < -slider_rect.height or slider_rect.y >= scaled_height:
                continue

            # Track
            pygame.draw.rect(popup_surface, (60, 60, 60), slider_rect, border_radius=3)

            # Filled portion
            ratio = ((element["value"] - element["min"])
                     / max(1e-9, element["max"] - element["min"]))
            value_w = int(ratio * element["rect"].width)
            pygame.draw.rect(
                popup_surface, self.button_color,
                (slider_rect.x, slider_rect.y, value_w, slider_rect.height),
                border_radius=3,
            )

            # Border
            pygame.draw.rect(
                popup_surface, self.border_color, slider_rect, 1, border_radius=3)

            # Knob
            knob_x = slider_rect.x + value_w
            knob_y = slider_rect.y + slider_rect.height // 2
            knob_r = int(slider_rect.height // 1.5)
            pygame.draw.circle(popup_surface, self.text_color, (knob_x, knob_y), knob_r)
            pygame.draw.circle(popup_surface, self.button_hover, (knob_x, knob_y), knob_r - 2)

    def _draw_buttons(self, popup_surface, font, sw, sh, sx, sy, y_offset):
        """Render action buttons and populate ``self.button_rects``."""
        self.button_rects = []
        if y_offset + 50 > sh + self.scroll_offset:
            return

        btn_y = max(sh - 50, y_offset + 10)
        total_w = sum(font.size(b["text"])[0] + 40 for b in self.buttons)
        total_w += (len(self.buttons) - 1) * 10
        btn_x = (sw - total_w) // 2

        for i, button in enumerate(self.buttons):
            bw = font.size(button["text"])[0] + 40
            rect = pygame.Rect(btn_x, btn_y, bw, 30)

            color = self.button_hover if i == self.hovered_button else self.button_color
            pygame.draw.rect(popup_surface, color, rect, border_radius=5)
            pygame.draw.rect(popup_surface, self.border_color, rect, 1, border_radius=5)

            txt = font.render(button["text"], True, self.text_color)
            popup_surface.blit(
                txt,
                (rect.x + (rect.width - txt.get_width()) // 2,
                 rect.y + (rect.height - txt.get_height()) // 2),
            )

            self.button_rects.append(
                pygame.Rect(sx + rect.x, sy + rect.y, rect.width, rect.height)
            )
            btn_x += bw + 10
    
    def is_visible(self):
        """Check if the pop-up is currently visible (not in closing animation)."""
        return self.visible and not self.closing


