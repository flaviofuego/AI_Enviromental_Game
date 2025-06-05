import pygame
import math
import random
from datetime import datetime
import time

from ..config.save_system import GameSaveSystem
from ..components.Button import Button
from ..components.LevelThumbnail import LevelThumbnail
from ..components.Card import Card
from ..components.AudioManager import audio_manager
from ..components.PopUp import PopUp, create_help_popup

class LevelSelectScreen:
    def __init__(self, save_system=None, screen=None):
        # Si no se pasa una ventana, crear una nueva (compatibilidad hacia atrás)
        if screen is None:
            pygame.init()
            # Configuración de pantalla adaptativa
            info = pygame.display.Info()
            self.screen_width = min(1200, info.current_w - 100)
            self.screen_height = min(800, info.current_h - 100)
            
            # Detectar si es formato móvil (relación de aspecto vertical)
            self.is_mobile = self.screen_height > self.screen_width
            
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Hockey Is Melting Down - Selección de Niveles")
        else:
            # Usar la ventana existente
            self.screen = screen
            self.screen_width = screen.get_width()
            self.screen_height = screen.get_height()
            self.is_mobile = self.screen_height > self.screen_width
        
        # Cargar imagen de fondo
        try:
            self.background_image = pygame.image.load('game/assets/niveles/background_levels.png')
            self.background_image = pygame.transform.scale(self.background_image, (self.screen_width, self.screen_height))
            self.background_opacity = 200  # 0-255, donde 255 es completamente opaco
        except (pygame.error, FileNotFoundError):
            print("No se pudo cargar el fondo del menú de niveles")
            self.background_image = None
        
        # Cargar sistema de guardado (o usar el proporcionado)
        if save_system:
            self.save_system = save_system
        else:
            print("Advertencia: No se proporcionó sistema de guardado. Creando uno nuevo.")
            self.save_system = GameSaveSystem()
        
        # Estado actual - Usar el perfil actual del sistema de guardado
        self.current_profile = self.save_system.current_profile
        if not self.current_profile:
            print("Error: No hay perfil activo en el sistema de guardado")
        self.selected_level = None
        self.selected_level_index = -1
        self.hover_level_index = -1
        self.show_level_info = False
        self.animate_selected = False
        self.animation_time = 0
        
        # Mensajes temporales
        self.message = ""
        self.message_time = 0
        self.message_type = "info"  # info, success, error
        
        # Colores temáticos (consistentes con home.py)
        self.colors = {
            'bg_gradient_top': (173, 216, 230),      # Azul espacial claro (hielo)
            'bg_gradient_bottom': (120, 50, 50),   # Rojo apocalíptico
            'ice_blue': (173, 216, 230),
            'critical_red': (220, 50, 50),
            'hope_green': (34, 139, 34),
            'warning_orange': (255, 140, 0),
            'text_white': (255, 255, 255),
            'text_gold': (255, 215, 0),
            'panel_dark': (20, 20, 40, 200),
            'button_active': (0, 100, 200),
            'button_hover': (0, 150, 255),
            'locked_gray': (140, 140, 140),
            'completed_green': (50, 180, 50),
            'level_card_bg': (40, 40, 60, 200)
        }
        
        # Fuentes
        try:
            self.font_title = pygame.font.Font(None, 48 if not self.is_mobile else 36)
            self.font_subtitle = pygame.font.Font(None, 24 if not self.is_mobile else 20)
            self.font_text = pygame.font.Font(None, 24 if not self.is_mobile else 20)
            self.font_small = pygame.font.Font(None, 20 if not self.is_mobile else 18)
        except:
            self.font_title = pygame.font.SysFont('Arial', 48 if not self.is_mobile else 36, bold=True)
            self.font_subtitle = pygame.font.SysFont('Arial', 24 if not self.is_mobile else 20, bold=True)
            self.font_text = pygame.font.SysFont('Arial', 24 if not self.is_mobile else 20)
            self.font_small = pygame.font.SysFont('Arial', 20 if not self.is_mobile else 18)
        
        # Definición de niveles
        self.levels = [
            {
                'id': 1,
                'name': "Basura en el Ártico",
                'enemy': "SLICKWAVE",
                'theme': "Plástico",
                'description': "Los mares del norte están inundados de plástico. Enfréntate a SlickWave, el emperador del plástico.",
                'challenge': "Recicla toneladas de desechos anotando goles",
                'icon': '🌊',
                'color': self.colors['ice_blue'],
                'unlocked': True,  # El primer nivel siempre está desbloqueado
                'completed': False
            },
            {
                'id': 2,
                'name': "Agujero de Ozono",
                'enemy': "UVBLADE",
                'theme': "Gases CFC",
                'description': "Los gases CFC han abierto un cráter en el cielo antártico. UVBlade controla la radiación.",
                'challenge': "Restaura el escudo protector anotando en las porterías moleculares",
                'icon': '☀️',
                'color': self.colors['warning_orange'],
                'unlocked': False,
                'completed': False
            },
            {
                'id': 3,
                'name': "Tormenta de Smog",
                'enemy': "SMOGATRON",
                'theme': "Aire Contaminado",
                'description': "La niebla tóxica asfixia las ciudades. Smogatron, el emperador del smog, bloquea el aire limpio.",
                'challenge': "Activa los filtros de aire con cada gol para purificar la atmósfera",
                'icon': '☁️',
                'color': (100, 100, 150),
                'unlocked': False,
                'completed': False
            },
            {
                'id': 4,
                'name': "Bosque Desvanecido",
                'enemy': "DEFORESTIX",
                'theme': "Deforestación",
                'description': "Los pulmones del planeta desaparecen rápidamente. Deforestix arrasa con todo a su paso.",
                'challenge': "Planta miles de árboles virtuales con cada victoria",
                'icon': '🌳',
                'color': self.colors['hope_green'],
                'unlocked': False,
                'completed': False
            },
            {
                'id': 5,
                'name': "Isla de Calor Urbano",
                'enemy': "HEATCORE",
                'theme': "Calentamiento Urbano",
                'description': "Las ciudades son hornos de asfalto. HeatCore eleva las temperaturas a niveles insoportables.",
                'challenge': "Enfría las ciudades con disparos certeros",
                'icon': '🔥',
                'color': self.colors['critical_red'],
                'unlocked': False,
                'completed': False
            }
        ]
        
        # Botones de navegación
        button_width = 150
        button_height = 40
        margin = 20
        
        self.buttons = {
            'back': {
                'rect': pygame.Rect(margin, self.screen_height - button_height - margin, button_width, button_height),
                'text': "Volver al Menú",
                'color': self.colors['ice_blue'],
                'hover_color': self.colors['button_hover']
            },
            'play': {
                'rect': pygame.Rect(self.screen_width - button_width - margin, 
                                   self.screen_height - button_height - margin, 
                                   button_width, button_height),
                'text': "¡Jugar Nivel!",
                'color': self.colors['hope_green'],
                'hover_color': (80, 200, 80)
            }
        }
        
        # Cargar efectos ambientales
        self.particles = []
        self.create_particles()
        
        # Inicializar reloj
        self.clock = pygame.time.Clock()
        
        # Actualizar estado de los niveles según perfil
        self.load_levels_status()
        
        # Inicializar cards
        self.cards = {}
        for level in self.levels:
            self.cards[level['id']] = Card(
                max_width=400 if not self.is_mobile else self.screen_width - 40,
                spacing=10,
                padding=10
            )
            
        # Inicializar miniaturas de niveles
        self.thumbnails = {}
        for level in self.levels:
            self.thumbnails[level['id']] = LevelThumbnail(level['id'])
        
        # Precargar audio para esta pantalla
        audio_manager.preload_audio_for_screen("level_select")

        # Inicializar el popup de ayuda
        self.help_popup = None
        
    def load_levels_status(self):
        """Actualizar el estado de los niveles según el perfil actual"""
        if self.current_profile:
            levels_unlocked = self.current_profile.get('levels', {}).get('unlocked', 1)
            completed_levels = self.current_profile.get('levels', {}).get('completed', [])
            
            for i, level in enumerate(self.levels):
                level['unlocked'] = level['id'] <= levels_unlocked
                level['completed'] = level['id'] in completed_levels
    
    def create_particles(self):
        """Crear partículas de fondo para el efecto ambiental"""
        for _ in range(30):
            self.particles.append({
                'x': random.randint(0, self.screen_width),
                'y': random.randint(0, self.screen_height),
                'size': random.randint(1, 3),
                'speed': random.uniform(0.5, 1.5),
                'angle': random.uniform(0, 2 * math.pi),
                'color': random.choice([
                    self.colors['ice_blue'],
                    self.colors['hope_green'],
                    self.colors['warning_orange'],
                    (100, 100, 150)  # Gris azulado
                ])
            })
    
    def update_particles(self):
        """Actualizar posición de las partículas"""
        for particle in self.particles:
            # Mover según ángulo y velocidad
            particle['x'] += math.cos(particle['angle']) * particle['speed']
            particle['y'] += math.sin(particle['angle']) * particle['speed']
            
            # Si sale de la pantalla, reiniciar en un borde aleatorio
            if (particle['x'] < 0 or particle['x'] > self.screen_width or
                particle['y'] < 0 or particle['y'] > self.screen_height):
                # Elegir un borde aleatorio (0=arriba, 1=derecha, 2=abajo, 3=izquierda)
                edge = random.randint(0, 3)
                
                if edge == 0:  # arriba
                    particle['x'] = random.randint(0, self.screen_width)
                    particle['y'] = 0
                elif edge == 1:  # derecha
                    particle['x'] = self.screen_width
                    particle['y'] = random.randint(0, self.screen_height)
                elif edge == 2:  # abajo
                    particle['x'] = random.randint(0, self.screen_width)
                    particle['y'] = self.screen_height
                else:  # izquierda
                    particle['x'] = 0
                    particle['y'] = random.randint(0, self.screen_height)
                
                # Nuevo ángulo para que vuelva al centro
                center_x, center_y = self.screen_width // 2, self.screen_height // 2
                particle['angle'] = math.atan2(center_y - particle['y'], center_x - particle['x'])
                particle['angle'] += random.uniform(-0.5, 0.5)  # Añadir algo de variación
    
    def draw_background(self):
        """Dibujar fondo con gradiente y efectos"""
        # Si hay imagen de fondo, dibujarla primero con transparencia
        if self.background_image:
            # Crear una copia de la imagen con la opacidad deseada
            temp_image = self.background_image.copy()
            temp_image.set_alpha(self.background_opacity)
            self.screen.blit(temp_image, (0, 0))
        else:
            # Dibujar gradiente como respaldo
            for y in range(self.screen_height):
                ratio = y / self.screen_height
                color = [
                    int(self.colors['bg_gradient_top'][i] * (1 - ratio) + 
                        self.colors['bg_gradient_bottom'][i] * ratio)
                    for i in range(3)
                ]
                pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))
        
        # Dibujar partículas sobre el fondo
        for particle in self.particles:
            particle['x'] += math.cos(particle['angle']) * particle['speed']
            particle['y'] += math.sin(particle['angle']) * particle['speed']
            
            # Wraparound
            if particle['x'] < 0:
                particle['x'] = self.screen_width
            elif particle['x'] > self.screen_width:
                particle['x'] = 0
            if particle['y'] < 0:
                particle['y'] = self.screen_height
            elif particle['y'] > self.screen_height:
                particle['y'] = 0
            
            # Dibujar partícula con efecto de brillo
            alpha = int(128 + 127 * math.sin(self.animation_time * 2 + particle['x'] * 0.01))
            pygame.draw.circle(self.screen, (*particle['color'], alpha), 
                             (int(particle['x']), int(particle['y'])), particle['size'])

    def draw_title(self):
        """Dibujar título de la pantalla de selección de niveles"""
        title_text = "SELECCIONA TU MISIÓN"
        
        # Posición adaptativa
        title_y = 50 if not self.is_mobile else 30
        
        # Efecto de brillo
        glow_intensity = abs(math.sin(self.animation_time * 3))
        
        # Sombra del título
        title_shadow = self.font_title.render(title_text, True, (0, 0, 0))
        shadow_rect = title_shadow.get_rect(center=(self.screen_width // 2 + 2, title_y + 2))
        self.screen.blit(title_shadow, shadow_rect)
        
        # Título con brillo variable
        title_color = (
            min(255, int(173 + 80 * glow_intensity)),
            min(255, int(216 + 40 * glow_intensity)),
            min(255, int(230 + 25 * glow_intensity))
        )
        
        title_surface = self.font_title.render(title_text, True, title_color)
        title_rect = title_surface.get_rect(center=(self.screen_width // 2, title_y))
        self.screen.blit(title_surface, title_rect)
        
        # Subtítulo
        subtitle_text = "Restaura el equilibrio climático, misión por misión"
        subtitle_surface = self.font_subtitle.render(subtitle_text, True, self.colors['text_gold'])
        subtitle_rect = subtitle_surface.get_rect(center=(self.screen_width // 2, title_y + 40))
        self.screen.blit(subtitle_surface, subtitle_rect)
        
        # Información del perfil
        if self.current_profile:
            profile_text = f"Agente: {self.current_profile['player_name']}"
            profile_surface = self.font_text.render(profile_text, True, self.colors['text_white'])
            self.screen.blit(profile_surface, (20, 20))
    
    def draw_level_cards(self):
        """Dibujar tarjetas de selección de nivel"""
        # Configuración de las tarjetas
        if self.is_mobile:
            cards_per_row = 1  # Móvil mantiene una columna
            gap = 20
            start_y = 150
        else:
            cards_per_row = 3  # Máximo de tarjetas por fila
            gap = 20
            start_y = 150
        
        # Calcular número de filas necesarias
        num_rows = math.ceil(len(self.levels) / cards_per_row)
        
        # Dibujar cada tarjeta de nivel
        for i, level in enumerate(self.levels):
            # Determinar estado y color del texto
            if not level['unlocked']:
                status_text = "BLOQUEADO"
                status_color = self.colors['locked_gray']
            elif level['completed']:
                status_text = "COMPLETADO"
                status_color = self.colors['completed_green']
            else:
                status_text = "DISPONIBLE"
                status_color = self.colors['ice_blue']
            
            # Calcular posición de la tarjeta
            if self.is_mobile:
                # Para móvil, una columna centrada
                card = self.cards[level['id']]
                card_rect = card.draw(
                    self.screen,
                    (20, start_y + (card.height + gap) * i),
                    self.thumbnails[level['id']].image,
                    status_text,
                    status_color,
                    i == self.selected_level_index,
                    abs(math.sin(self.animation_time * 10)) if i == self.selected_level_index else 0
                )
            else:
                # Para PC, distribuir en filas
                row = i // cards_per_row
                col = i % cards_per_row
                
                # Dibujar la tarjeta
                card = self.cards[level['id']]
                
                # Si es la última fila y no está completa, centrar las tarjetas
                if row == num_rows - 1:
                    cards_in_last_row = len(self.levels) - (row * cards_per_row)
                    if cards_in_last_row < cards_per_row:
                        total_width = (card.width + gap) * cards_in_last_row - gap
                        start_x = (self.screen_width - total_width) // 2
                    else:
                        total_width = (card.width + gap) * cards_per_row - gap
                        start_x = (self.screen_width - total_width) // 2
                else:
                    total_width = (card.width + gap) * cards_per_row - gap
                    start_x = (self.screen_width - total_width) // 2
                
                x = start_x + (card.width + gap) * col
                y = start_y + (card.height + gap) * row
                
                card_rect = card.draw(
                    self.screen,
                    (x, y),
                    self.thumbnails[level['id']].image,
                    status_text,
                    status_color,
                    i == self.selected_level_index,
                    abs(math.sin(self.animation_time * 10)) if i == self.selected_level_index else 0
                )
            
            # Guardar rectángulo para detección de clics
            level['rect'] = card_rect
    
    def draw_level_details(self):
        """Dibujar panel de detalles del nivel seleccionado"""
        if self.selected_level_index < 0 or not self.show_level_info:
            return
            
        level = self.levels[self.selected_level_index]
        
        # Configurar panel de información
        panel_width = 500 if not self.is_mobile else self.screen_width - 40
        panel_height = 250
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = self.screen_height - panel_height - 80
        
        # Crear panel semi-transparente
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((20, 20, 40, 220))
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Borde con el color del nivel
        pygame.draw.rect(self.screen, level['color'], (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Título del nivel
        title_text = f"Nivel {level['id']}: {level['name']}"
        title_surface = self.font_subtitle.render(title_text, True, level['color'])
        self.screen.blit(title_surface, (panel_x + 20, panel_y + 20))
        
        # Enemigo
        enemy_text = f"Enemigo: {level['enemy']} - {level['theme']}"
        enemy_surface = self.font_text.render(enemy_text, True, self.colors['text_gold'])
        self.screen.blit(enemy_surface, (panel_x + 20, panel_y + 50))
        
        # Descripción
        desc_lines = self.wrap_text(level['description'], self.font_small, panel_width - 40)
        y_offset = 80
        for line in desc_lines:
            text_surface = self.font_small.render(line, True, self.colors['text_white'])
            self.screen.blit(text_surface, (panel_x + 20, panel_y + y_offset))
            y_offset += 20
        
        # Desafío
        challenge_lines = self.wrap_text("Desafío: " + level['challenge'], self.font_small, panel_width - 40)
        y_offset += 10
        for line in challenge_lines:
            text_surface = self.font_small.render(line, True, self.colors['hope_green'])
            self.screen.blit(text_surface, (panel_x + 20, panel_y + y_offset))
            y_offset += 20
        
        # Estado
        if not level['unlocked']:
            status_text = "¡Completa el nivel anterior para desbloquear este!"
            status_color = self.colors['locked_gray']
        elif level['completed']:
            status_text = "¡Nivel completado! Puedes volver a jugarlo para mejorar tu puntuación."
            status_color = self.colors['completed_green']
        else:
            status_text = "¡Nivel listo para jugar! Ayuda a restaurar el planeta."
            status_color = self.colors['ice_blue']
            
        status_surface = self.font_text.render(status_text, True, status_color)
        self.screen.blit(status_surface, (panel_x + 20, panel_y + panel_height - 40))
    
    def draw_buttons(self):
        """Dibujar botones de navegación"""
        for key, button in self.buttons.items():
            # Ver si el mouse está sobre el botón
            mouse_pos = pygame.mouse.get_pos()
            is_hover = button['rect'].collidepoint(mouse_pos)
            
            # Elegir color según estado
            color = button['hover_color'] if is_hover else button['color']
            
            # Dibujar botón
            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, self.colors['text_white'], button['rect'], 2)
            
            # Texto del botón
            text_surface = self.font_text.render(button['text'], True, self.colors['text_white'])
            text_rect = text_surface.get_rect(center=button['rect'].center)
            self.screen.blit(text_surface, text_rect)
            
            # Deshabilitar botón "Jugar" si no hay nivel seleccionado o está bloqueado
            if key == 'play':
                if (self.selected_level_index < 0 or 
                    not self.levels[self.selected_level_index]['unlocked']):
                    # Dibujar overlay semi-transparente para indicar deshabilitado
                    disabled_overlay = pygame.Surface(button['rect'].size, pygame.SRCALPHA)
                    disabled_overlay.fill((0, 0, 0, 128))
                    self.screen.blit(disabled_overlay, button['rect'])
    
    def draw_message(self):
        """Dibujar mensaje temporal si existe"""
        if self.message and time.time() - self.message_time < 3:
            # Elegir color según tipo de mensaje
            if self.message_type == "success":
                color = self.colors['hope_green']
            elif self.message_type == "error":
                color = self.colors['critical_red']
            else:
                color = self.colors['ice_blue']
                
            # Crear mensaje
            message_surface = self.font_text.render(self.message, True, color)
            message_rect = message_surface.get_rect(center=(self.screen_width // 2, self.screen_height - 20))
            
            # Fondo del mensaje
            bg_rect = message_rect.inflate(40, 10)
            bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surface.fill((20, 20, 40, 200))
            self.screen.blit(bg_surface, bg_rect)
            pygame.draw.rect(self.screen, color, bg_rect, 1)
            
            # Dibujar texto
            self.screen.blit(message_surface, message_rect)
    
    def wrap_text(self, text, font, max_width):
        """Partir texto en múltiples líneas para ajustar a un ancho máximo"""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            # Probar añadir esta palabra a la línea actual
            test_line = ' '.join(current_line + [word])
            width, _ = font.size(test_line)
            
            if width <= max_width:
                current_line.append(word)
            else:
                # Si no cabe, comenzar nueva línea
                lines.append(' '.join(current_line))
                current_line = [word]
        
        # Añadir la última línea
        if current_line:
            lines.append(' '.join(current_line))
            
        return lines
    
    def show_message(self, message, message_type="info"):
        """Mostrar un mensaje temporal en pantalla"""
        self.message = message
        self.message_time = time.time()
        self.message_type = message_type
    
    def handle_click(self, pos):
        """Manejar clics del mouse"""
        # Comprobar clics en niveles
        for i, level in enumerate(self.levels):
            if 'rect' in level and level['rect'].collidepoint(pos):
                if level['unlocked']:
                    # Reproducir sonido de selección
                    audio_manager.play_sound_effect("button_click")
                    
                    self.selected_level_index = i
                    self.selected_level = level
                    self.show_level_info = True
                    self.animate_selected = True
                    return True
                else:
                    # Sonido de error para nivel bloqueado
                    audio_manager.play_sound_effect("button_click", volume_override=0.2)
                    self.show_message("Este nivel está bloqueado. ¡Completa los niveles anteriores primero!", "error")
                    return False
        
        # Comprobar clics en botones
        for key, button in self.buttons.items():
            if button['rect'].collidepoint(pos):
                # Reproducir sonido de botón
                audio_manager.play_sound_effect("button_click")
                
                if key == 'back':
                    # Volver al menú principal
                    return 'back_to_menu'
                elif key == 'play':
                    # Verificar que hay nivel seleccionado y está desbloqueado
                    if (self.selected_level_index >= 0 and 
                        self.levels[self.selected_level_index]['unlocked']):
                        level_id = self.levels[self.selected_level_index]['id']
                        return f'start_level_{level_id}'
                    else:
                        self.show_message("Selecciona un nivel disponible para jugar", "error")
                        return False
                elif key == 'help':
                    # Mostrar popup de ayuda
                    self.help_popup = create_help_popup(self.screen, "level_select")
                    self.help_popup.show()
                    return False
        
        # Clic en espacio vacío
        self.show_level_info = False
        self.animate_selected = False
        return False
    
    def handle_hover(self, pos):
        """Manejar hover sobre elementos"""
        # Reset hover state
        old_hover = self.hover_level_index
        self.hover_level_index = -1
        
        # Check hover over levels
        for i, level in enumerate(self.levels):
            if 'rect' in level and level['rect'].collidepoint(pos):
                self.hover_level_index = i
                # Reproducir sonido de hover si cambió
                if old_hover != i and level['unlocked']:
                    audio_manager.play_sound_effect("button_hover", volume_override=0.2)
                return
    
    def run(self):
        """Bucle principal"""
        running = True
        result = None

        help_button = Button('help', (40, 40), (self.screen_width - 40, 40), "Ayuda")

        while running:
            dt = self.clock.tick(60) / 1000.0  # Delta time en segundos
            self.animation_time += dt
            
            # Actualizar partículas
            self.update_particles()

            # Actualizar popup si existe
            if self.help_popup:
                popup_result = self.help_popup.update(dt)
                if popup_result == "closed":
                    self.help_popup = None
            
            # Procesar eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "exit"  # Retornar "exit" directamente en lugar de volver al menú
                
                # Manejar eventos del popup si está visible
                if self.help_popup and self.help_popup.is_visible():
                    popup_action = self.help_popup.handle_event(event)
                    if popup_action == "more_info":
                        # Aquí podrías implementar alguna acción adicional
                        pass
                    continue  # Si el popup está visible, no procesar otros eventos
                    

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Clic izquierdo
                        action = self.handle_click(event.pos)
                        if action and isinstance(action, str):
                            result = action
                            running = False
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_hover(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "exit"  # Salir del juego con la tecla Escape
            
            # Dibujar pantalla
            self.draw_background()
            self.draw_title()
            self.draw_level_cards()
            if self.show_level_info:
                self.draw_level_details()
            self.draw_buttons()
            help_button.draw(self.screen)
            if help_button.is_clicked(pygame.mouse.get_pos()):
                self.help_popup = create_help_popup(self.screen, "level_select")
                self.help_popup.show()
            self.draw_message()

            # Dibujar popup de ayuda si está visible
            if self.help_popup:
                self.help_popup.draw()
            
            # Actualizar pantalla
            pygame.display.flip()
        
        return result

# Función para ejecutar la pantalla de forma independiente (pruebas)
if __name__ == "__main__":
    screen = LevelSelectScreen()
    result = screen.run()
    print(f"Result: {result}")