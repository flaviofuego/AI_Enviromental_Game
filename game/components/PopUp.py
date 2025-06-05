import pygame
import math
import time

class PopUp:
    """
    Componente reutilizable para crear ventanas emergentes (pop-ups) 
    con contenido personalizable, animaciones y efectos visuales.
    """
    
    def __init__(self, screen, title="", content=[], buttons=None, popup_type="info"):
        """
        Inicializar el pop-up
        
        Args:
            screen: Superficie de pygame donde dibujar
            title: Título del pop-up
            content: Lista de líneas de texto para el contenido
            buttons: Lista de diccionarios con botones {"text": "OK", "action": "close"}
            popup_type: Tipo de pop-up ("info", "warning", "error", "help")
        """
        self.screen = screen
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()
        
        self.title = title
        self.content = content if isinstance(content, list) else [content]
        self.popup_type = popup_type
        self.visible = False
        self.closing = False
        
        # Configuración de botones por defecto
        self.buttons = buttons if buttons else [{"text": "OK", "action": "close"}]
        
        # Configuración visual según tipo
        self.setup_visual_config()
        
        # Configuración de animación
        self.animation_time = 0
        self.fade_duration = 0.3
        self.scale_animation = True
        self.current_scale = 0.0
        self.target_scale = 1.0
        
        # Estado de hover/click
        self.hovered_button = -1
        self.button_rects = []

        # Agregar soporte para elementos interactivos
        self.interactive_elements = []
        self.active_element = None
        
        # Fuentes
        try:
            self.font_title = pygame.font.Font(None, 28)
            self.font_content = pygame.font.Font(None, 20)
            self.font_button = pygame.font.Font(None, 18)
        except:
            self.font_title = pygame.font.SysFont('Arial', 28, bold=True)
            self.font_content = pygame.font.SysFont('Arial', 20)
            self.font_button = pygame.font.SysFont('Arial', 18)
        
        # Calcular dimensiones
        self.calculate_dimensions()

    def add_slider(self, x, y, width, height, min_value, max_value, current_value, id):
        """Añadir un slider interactivo"""
        slider = {
            "type": "slider",
            "rect": pygame.Rect(x, y, width, height),
            "min": min_value,
            "max": max_value,
            "value": current_value,
            "id": id,
            "dragging": False
        }
        self.interactive_elements.append(slider)
        return slider
    
    def setup_visual_config(self):
        """Configurar colores y estilos según el tipo de pop-up"""
        configs = {
            "info": {
                "bg_color": (20, 30, 50, 240),
                "border_color": (100, 150, 200),
                "title_color": (173, 216, 230),
                "text_color": (255, 255, 255),
                "button_color": (60, 100, 140),
                "button_hover": (80, 120, 160),
                "icon": "ℹ️"
            },
            "warning": {
                "bg_color": (50, 40, 20, 240),
                "border_color": (255, 200, 100),
                "title_color": (255, 215, 0),
                "text_color": (255, 255, 255),
                "button_color": (200, 140, 60),
                "button_hover": (220, 160, 80),
                "icon": "⚠️"
            },
            "error": {
                "bg_color": (50, 20, 20, 240),
                "border_color": (220, 100, 100),
                "title_color": (255, 100, 100),
                "text_color": (255, 255, 255),
                "button_color": (180, 60, 60),
                "button_hover": (200, 80, 80),
                "icon": "❌"
            },
            "help": {
                "bg_color": (20, 40, 30, 240),
                "border_color": (100, 200, 150),
                "title_color": (100, 255, 150),
                "text_color": (240, 255, 240),
                "button_color": (60, 140, 100),
                "button_hover": (80, 160, 120),
                "icon": "❓"
            }
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
        """Manejar eventos del pop-up"""
        if not self.visible or self.closing:
            return None
        
        result = None
        
        if event.type == pygame.MOUSEMOTION:
            result = self.handle_mouse_motion(event.pos)
            # Actualizar sliders si están siendo arrastrados
            for element in self.interactive_elements:
                if element["type"] == "slider" and element["dragging"]:
                    relative_x = max(0, min(event.pos[0] - (self.x + element["rect"].x), element["rect"].width))
                    element["value"] = element["min"] + (element["max"] - element["min"]) * (relative_x / element["rect"].width)
                    return {"action": "slider_change", "id": element["id"], "value": element["value"]}
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Click izquierdo
                # Comprobar si se hizo clic en algún elemento interactivo
                real_pos = event.pos
                for element in self.interactive_elements:
                    # Ajustar la posición del elemento a la pantalla
                    element_rect = pygame.Rect(self.x + element["rect"].x, 
                                              self.y + element["rect"].y,
                                              element["rect"].width, 
                                              element["rect"].height)
                    
                    if element_rect.collidepoint(real_pos):
                        if element["type"] == "slider":
                            element["dragging"] = True
                            relative_x = max(0, min(real_pos[0] - element_rect.x, element_rect.width))
                            element["value"] = element["min"] + (element["max"] - element["min"]) * (relative_x / element_rect.width)
                            return {"action": "slider_change", "id": element["id"], "value": element["value"]}
                
                # Si no se hizo clic en un elemento interactivo, comprobar botones
                btn_result = self.handle_click(event.pos)
                if btn_result:
                    return btn_result
            elif event.button == 4:  # Scroll up
                self.scroll_up()
            elif event.button == 5:  # Scroll down
                self.scroll_down()
        elif event.type == pygame.MOUSEBUTTONUP:
            # Detener arrastre de elementos
            for element in self.interactive_elements:
                if element["type"] == "slider" and element["dragging"]:
                    element["dragging"] = False
                    return {"action": "slider_final", "id": element["id"], "value": element["value"]}
        
        
        if event.type == pygame.MOUSEMOTION:
            return self.handle_mouse_motion(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Click izquierdo
                return self.handle_click(event.pos)
            elif event.button == 4:  # Scroll up
                self.scroll_up()
            elif event.button == 5:  # Scroll down
                self.scroll_down()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.close()
                return "close"
            elif event.key == pygame.K_RETURN:
                # Presionar el primer botón
                if self.buttons:
                    return self.buttons[0]["action"]
            elif event.key == pygame.K_UP:
                self.scroll_up()
            elif event.key == pygame.K_DOWN:
                self.scroll_down()
        
        return result
    
    def handle_mouse_motion(self, pos):
        """Manejar movimiento del mouse para hover effects"""
        old_hover = self.hovered_button
        self.hovered_button = -1
        
        # Verificar hover en botones
        for i, rect in enumerate(self.button_rects):
            if rect.collidepoint(pos):
                self.hovered_button = i
                break
        
        # Reproducir sonido de hover si cambió
        if old_hover != self.hovered_button and self.hovered_button != -1:
            from .AudioManager import audio_manager
            audio_manager.play_sound_effect("button_hover", volume_override=0.1)
        
        return None
    
    def handle_click(self, pos):
        """Manejar clicks en botones"""
        # Verificar click en botones
        for i, rect in enumerate(self.button_rects):
            if rect.collidepoint(pos):
                from .AudioManager import audio_manager
                audio_manager.play_sound_effect("button_click", volume_override=0.3)
                
                action = self.buttons[i].get("action", "close")
                if action == "close":
                    self.close()
                return action
        
        # Click fuera del popup lo cierra (opcional)
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
        """Dibujar el pop-up"""
        if not self.visible:
            return
        
        # Overlay semi-transparente
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))
        
        # Calcular posición y tamaño con escala
        scaled_width = int(self.width * self.current_scale)
        scaled_height = int(self.height * self.current_scale)
        scaled_x = self.x + (self.width - scaled_width) // 2
        scaled_y = self.y + (self.height - scaled_height) // 2
        
        if scaled_width <= 0 or scaled_height <= 0:
            return
        
        # Crear superficie del pop-up
        popup_surface = pygame.Surface((scaled_width, scaled_height), pygame.SRCALPHA)
        
        # Fondo del pop-up
        popup_surface.fill(self.bg_color)
        
        # Borde con brillo
        border_width = 2
        glow_intensity = abs(math.sin(self.animation_time * 3)) * 0.3 + 0.7
        glow_color = tuple(min(255, int(c * glow_intensity)) for c in self.border_color[:3])
        pygame.draw.rect(popup_surface, glow_color, 
                        (0, 0, scaled_width, scaled_height), border_width)
        
        # Escalar fuentes proporcionalmente
        if self.current_scale < 1.0:
            # Durante la animación, escalar el contenido
            scale_factor = max(0.1, self.current_scale)
            font_title_size = max(12, int(28 * scale_factor))
            font_content_size = max(10, int(20 * scale_factor))
            font_button_size = max(8, int(18 * scale_factor))
            
            try:
                scaled_font_title = pygame.font.Font(None, font_title_size)
                scaled_font_content = pygame.font.Font(None, font_content_size)
                scaled_font_button = pygame.font.Font(None, font_button_size)
            except:
                scaled_font_title = pygame.font.SysFont('Arial', font_title_size, bold=True)
                scaled_font_content = pygame.font.SysFont('Arial', font_content_size)
                scaled_font_button = pygame.font.SysFont('Arial', font_button_size)
        else:
            scaled_font_title = self.font_title
            scaled_font_content = self.font_content
            scaled_font_button = self.font_button
        
        # Contenido (solo si está suficientemente escalado)
        if self.current_scale > 0.3:
            # Dibujar elementos interactivos
            for element in self.interactive_elements:
                if element["type"] == "slider":
                    # Dibujar slider base
                    slider_rect = pygame.Rect(element["rect"].x, 
                                             max(0, element["rect"].y - self.scroll_offset), 
                                             element["rect"].width, 
                                             element["rect"].height)
                    
                    # Solo dibujar si está visible
                    if slider_rect.y > -slider_rect.height and slider_rect.y < scaled_height:
                        # Barra de fondo
                        pygame.draw.rect(popup_surface, (60, 60, 60), slider_rect, border_radius=3)
                        
                        # Barra de valor
                        value_width = int((element["value"] - element["min"]) / 
                                         (element["max"] - element["min"]) * element["rect"].width)
                        value_rect = pygame.Rect(slider_rect.x, slider_rect.y, value_width, slider_rect.height)
                        pygame.draw.rect(popup_surface, self.button_color, value_rect, border_radius=3)
                        
                        # Marco
                        pygame.draw.rect(popup_surface, self.border_color, slider_rect, 1, border_radius=3)
                        
                        # Círculo indicador
                        knob_x = slider_rect.x + value_width
                        knob_y = slider_rect.y + slider_rect.height // 2
                        knob_radius = slider_rect.height // 1.5
                        pygame.draw.circle(popup_surface, self.text_color, (knob_x, knob_y), knob_radius)
                        pygame.draw.circle(popup_surface, self.button_hover, (knob_x, knob_y), knob_radius - 2)

            y_offset = 20 - self.scroll_offset

            # Título con icono
            if self.title:
                title_text = f"{self.title}"
                title_surface = scaled_font_title.render(title_text, True, self.title_color)
                title_x = (scaled_width - title_surface.get_width()) // 2
                if y_offset > -title_surface.get_height() and y_offset < scaled_height:
                    popup_surface.blit(title_surface, (title_x, max(0, y_offset)))
                y_offset += 40
            
            # Línea separadora
            if self.title and y_offset > 0 and y_offset < scaled_height:
                line_y = min(y_offset, scaled_height - 5)
                pygame.draw.line(popup_surface, self.border_color, 
                               (20, line_y), (scaled_width - 20, line_y))
            y_offset += 10
            
            # Contenido
            for line in self.content:
                if y_offset > scaled_height:
                    break
                if y_offset > -25:
                    content_surface = scaled_font_content.render(line, True, self.text_color)
                    popup_surface.blit(content_surface, (20, max(0, y_offset)))
                y_offset += 25
            
            # Botones
            self.button_rects = []
            if y_offset + 50 <= scaled_height + self.scroll_offset:
                button_y = max(scaled_height - 50, y_offset + 10)
                total_button_width = sum(scaled_font_button.size(btn["text"])[0] + 40 for btn in self.buttons)
                total_button_width += (len(self.buttons) - 1) * 10
                
                button_x = (scaled_width - total_button_width) // 2
                
                for i, button in enumerate(self.buttons):
                    button_width = scaled_font_button.size(button["text"])[0] + 40
                    button_rect = pygame.Rect(button_x, button_y, button_width, 30)
                    
                    # Color del botón
                    if i == self.hovered_button:
                        button_color = self.button_hover
                    else:
                        button_color = self.button_color
                    
                    # Dibujar botón
                    pygame.draw.rect(popup_surface, button_color, button_rect, border_radius=5)
                    pygame.draw.rect(popup_surface, self.border_color, button_rect, 1, border_radius=5)
                    
                    # Texto del botón
                    button_text = scaled_font_button.render(button["text"], True, self.text_color)
                    text_x = button_rect.x + (button_rect.width - button_text.get_width()) // 2
                    text_y = button_rect.y + (button_rect.height - button_text.get_height()) // 2
                    popup_surface.blit(button_text, (text_x, text_y))
                    
                    # Guardar rectángulo ajustado a la posición real
                    real_rect = pygame.Rect(scaled_x + button_rect.x, scaled_y + button_rect.y,
                                          button_rect.width, button_rect.height)
                    self.button_rects.append(real_rect)
                    
                    button_x += button_width + 10
            
            # Indicador de scroll si es necesario
            if self.needs_scroll and self.current_scale > 0.8:
                if self.scroll_offset > 0:
                    # Flecha hacia arriba
                    pygame.draw.polygon(popup_surface, self.text_color, 
                                      [(scaled_width - 20, 10), (scaled_width - 15, 5), (scaled_width - 10, 10)])
                
                if self.scroll_offset < self.max_scroll:
                    # Flecha hacia abajo
                    pygame.draw.polygon(popup_surface, self.text_color,
                                      [(scaled_width - 20, scaled_height - 10), 
                                       (scaled_width - 15, scaled_height - 5), 
                                       (scaled_width - 10, scaled_height - 10)])
        
        # Dibujar el pop-up en la pantalla
        self.screen.blit(popup_surface, (scaled_x, scaled_y))
    
    def is_visible(self):
        """Verificar si el pop-up está visible"""
        return self.visible and not self.closing

def create_help_popup(screen, screen_name):
    """
    Crear pop-up de ayuda contextual según la pantalla actual
    
    Args:
        screen: Superficie de pygame
        screen_name: Nombre de la pantalla ("home", "level_select", etc.)
    """
    help_content = {
        "home": {
            "title": "Menú Principal - Ayuda",
            "content": [
                "CONTROLES PRINCIPALES:",
                "",
                "• Clic en JUGAR: Accede a la selección de niveles",
                "• Clic en HISTORIAL: Muestra información de GAIA",
                "• Clic en JUGADOR: Gestiona perfiles de jugador",
                "• Clic en CONFIGURACIÓN: Ajusta opciones del juego",
                "",
                "PANEL DE PROGRESO:",
                "• Muestra tu avance planetario en tiempo real",
                "• Océanos, ozono, aire, bosques y ciudades",
                "• Puntos GAIA acumulados por tus acciones",
                "",
                "MISIÓN:",
                "• La Tierra está en crisis climática",
                "• 5 agentes de EcoNull controlan el clima",
                "• Derrota a cada uno para restaurar el planeta",
                "",
                "CONSEJOS:",
                "• Crea o selecciona un perfil antes de jugar",
                "• Revisa el historial para entender la crisis",
                "• Tu progreso se guarda automáticamente"
            ]
        },
        "level_select": {
            "title": "Selección de Niveles - Ayuda", 
            "content": [
                "SELECCIÓN DE MISIONES:",
                "",
                "• Clic en cualquier nivel para ver detalles",
                "• Los niveles se desbloquean progresivamente",
                "• Completa un nivel para acceder al siguiente",
                "",
                "ESTADOS DE NIVEL:",
                "• DISPONIBLE (azul): Listo para jugar",
                "• COMPLETADO (verde): Ya derrotaste al enemigo",
                "• BLOQUEADO (gris): Necesitas completar niveles anteriores",
                "",
                "ENEMIGOS DE ECONULL:",
                "• SLICKWAVE: Emperador del plástico oceánico",
                "• UVBLADE: Destructor del escudo de ozono",
                "• SMOGATRON: Señor de la contaminación atmosférica",
                "• DEFORESTIX: Talador de los bosques planetarios",
                "• HEATCORE: Maestro de las islas de calor urbano",
                "",
                "CONTROLES:",
                "• Selecciona un nivel y presiona '¡Jugar Nivel!'",
                "• 'Volver al Menú' regresa a la pantalla principal",
                "• ESC para salir del juego en cualquier momento",
                "",
                "ESTRATEGIA:",
                "• Lee la descripción de cada misión",
                "• Cada enemigo tiene mecánicas únicas",
                "• Puedes repetir niveles para mejorar puntuación"
            ]
        }
    }
    
    content_data = help_content.get(screen_name, {
        "title": "Ayuda General",
        "content": [
            "Esta es la ventana de ayuda general.",
            "",
            "Usa los controles del mouse para navegar.",
            "Presiona ESC para cerrar ventanas.",
            "Tu progreso se guarda automáticamente."
        ]
    })
    
    buttons = [
        {"text": "Entendido", "action": "close"},
        {"text": "Más Info", "action": "more_info"}
    ]
    
    return PopUp(screen, content_data["title"], content_data["content"], buttons, "help")


def create_settings_popup(screen, audio_manager):
    """
    Crear pop-up de configuración con controles interactivos para el audio
    
    Args:
        screen: Superficie de pygame
        audio_manager: Instancia del administrador de audio
    """
    # Obtener valores actuales directamente usando los nuevos métodos
    music_volume = audio_manager.get_music_volume()
    sfx_volume = audio_manager.get_sfx_volume()
    music_enabled = audio_manager.is_music_enabled()
    sfx_enabled = audio_manager.is_sfx_enabled()

    # Textos de botones según estado actual
    music_button_text = "Música: Activada" if music_enabled else "Música: Desactivada"
    sfx_button_text = "Efectos: Activados" if sfx_enabled else "Efectos: Desactivados"
    
    settings_content = [
        "Ajusta las opciones del juego:",
        "",
        "• Volumen de Música:",
        "",  # Espacio para el slider
        "",
        "• Volumen de Efectos:",
        "",  # Espacio para el slider
    ]

    buttons = [
        {"text": music_button_text, "action": "toggle_music"},
        {"text": sfx_button_text, "action": "toggle_sfx"},
        {"text": "Guardar", "action": "save_settings"},
        {"text": "Cancelar", "action": "cancel"}
    ]

    popup = PopUp(screen, "Configuración", settings_content, buttons, "info")
    
    # Añadir sliders después de calculadas las dimensiones
    # Los valores de posición (x, y) son relativos al popup
    slider_width = int(popup.width * 0.7)
    slider_x = (popup.width - slider_width) // 2
    
    # Slider para música - ajustar posición vertical según contenido
    music_y = 90  # Posición aproximada debajo del texto "Volumen de Música:"
    popup.add_slider(slider_x, music_y, slider_width, 10, 0.0, 1.0, music_volume, "music_volume")
    
    # Slider para efectos - ajustar posición vertical según contenido
    sfx_y = 150  # Posición aproximada debajo del texto "Volumen de Efectos:"
    popup.add_slider(slider_x, sfx_y, slider_width, 10, 0.0, 1.0, sfx_volume, "sfx_volume")
    
    return popup