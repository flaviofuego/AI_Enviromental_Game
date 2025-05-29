import pygame
import math
import random
from datetime import datetime
import time

from ..config.save_system import GameSaveSystem
from ..components.Button import Button

class LevelSelectScreen:
    def __init__(self, save_system=None):
        pygame.init()
        
        # Configuración de pantalla adaptativa
        info = pygame.display.Info()
        self.screen_width = min(1200, info.current_w - 100)
        self.screen_height = min(800, info.current_h - 100)
        
        # Detectar si es formato móvil (relación de aspecto vertical)
        self.is_mobile = self.screen_height > self.screen_width
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Hockey Is Melting Down - Selección de Niveles")
        
        # Cargar sistema de guardado (o usar el proporcionado)
        if save_system:
            self.save_system = save_system
        else:
            self.save_system = GameSaveSystem()
        
        # Estado actual
        self.current_profile = self.save_system.current_profile
        self.selected_level = None
        self.selected_level_index = -1
        self.hover_level_index = -1
        self.show_level_info = False
        self.animate_selected = False
        self.animation_time = 0
        
        # Variables para efectos de transición
        self.transitioning = False
        self.transition_alpha = 0
        self.transition_direction = 1  # 1 para fade in, -1 para fade out
        
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
            'locked_gray': (100, 100, 100),
            'completed_green': (50, 180, 50),
            'level_card_bg': (40, 40, 60, 200)
        }
        
        # Fuentes
        try:
            self.font_title = pygame.font.Font(None, 48 if not self.is_mobile else 36)
            self.font_subtitle = pygame.font.Font(None, 24 if not self.is_mobile else 20)
            self.font_text = pygame.font.Font(None, 18 if not self.is_mobile else 16)
            self.font_small = pygame.font.Font(None, 14 if not self.is_mobile else 12)
        except:
            self.font_title = pygame.font.SysFont('Arial', 48 if not self.is_mobile else 36, bold=True)
            self.font_subtitle = pygame.font.SysFont('Arial', 24 if not self.is_mobile else 20, bold=True)
            self.font_text = pygame.font.SysFont('Arial', 18 if not self.is_mobile else 16)
            self.font_small = pygame.font.SysFont('Arial', 14 if not self.is_mobile else 12)
        
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
        """Dibujar fondo con gradiente y partículas"""
        # Gradiente de fondo
        for y in range(self.screen_height):
            ratio = y / self.screen_height
            color = [
                int(self.colors['bg_gradient_top'][i] * (1 - ratio) + 
                    self.colors['bg_gradient_bottom'][i] * ratio)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))
        
        # Dibujar partículas
        for particle in self.particles:
            # Tamaño variable con pulsación
            size_mod = math.sin(self.animation_time + particle['x'] * 0.01) * 0.5 + 1.5
            size = particle['size'] * size_mod
            
            pygame.draw.circle(
                self.screen,
                particle['color'],
                (int(particle['x']), int(particle['y'])),
                int(size)
            )

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
            # Disposición vertical para móvil
            card_width = self.screen_width - 40
            card_height = 80
            start_x = 20
            start_y = 120
            gap = 20
        else:
            # Disposición horizontal para PC
            card_width = 180
            card_height = 120
            total_width = (card_width + 20) * len(self.levels) - 20
            start_x = (self.screen_width - total_width) // 2
            start_y = 120
            gap = 20
        
        # Crear un rectángulo para contener todas las tarjetas
        container_width = self.screen_width - 40 if self.is_mobile else total_width + 40
        container_height = (card_height + gap) * len(self.levels) if self.is_mobile else card_height + 40
        container_x = (self.screen_width - container_width) // 2
        container_y = start_y - 20
        
        # Dibujar contenedor
        container_surface = pygame.Surface((container_width, container_height), pygame.SRCALPHA)
        container_surface.fill((20, 20, 40, 150))
        self.screen.blit(container_surface, (container_x, container_y))
        pygame.draw.rect(self.screen, self.colors['ice_blue'], 
                        (container_x, container_y, container_width, container_height), 2)
        
        # Dibujar cada tarjeta de nivel
        for i, level in enumerate(self.levels):
            # Calcular posición basada en índice y diseño
            if self.is_mobile:
                x = start_x
                y = start_y + (card_height + gap) * i
            else:
                x = start_x + (card_width + gap) * i
                y = start_y
            
            # Comprobar si está seleccionado o con hover
            is_selected = i == self.selected_level_index
            is_hover = i == self.hover_level_index and level['unlocked']
            
            # Determinar color basado en estado
            if not level['unlocked']:
                card_color = self.colors['locked_gray']
                border_color = self.colors['locked_gray']
            elif is_selected or is_hover:
                # Animación pulsante para el seleccionado
                if is_selected and self.animate_selected:
                    glow = abs(math.sin(self.animation_time * 10)) * 50
                    card_color = level['color']
                    border_color = (min(255, level['color'][0] + int(glow)),
                                    min(255, level['color'][1] + int(glow)),
                                    min(255, level['color'][2] + int(glow)))
                else:
                    card_color = level['color']
                    border_color = self.colors['text_gold']
            elif level['completed']:
                card_color = self.colors['completed_green']
                border_color = self.colors['text_gold']
            else:
                card_color = level['color']
                border_color = self.colors['ice_blue']
            
            # Dibujar tarjeta
            card_rect = pygame.Rect(x, y, card_width, card_height)
            pygame.draw.rect(self.screen, self.colors['level_card_bg'], card_rect)
            pygame.draw.rect(self.screen, border_color, card_rect, 2)
            
            # Colorear borde superior para indicar tema
            theme_bar_rect = pygame.Rect(x+2, y+2, card_width-4, 10)
            pygame.draw.rect(self.screen, card_color, theme_bar_rect)
            
            # Icono del nivel
            icon_x = x + 20
            icon_y = y + card_height // 2
            icon_surface = self.font_subtitle.render(level['icon'], True, card_color)
            icon_rect = icon_surface.get_rect(center=(icon_x, icon_y))
            self.screen.blit(icon_surface, icon_rect)
            
            # Textos de nivel
            level_num = f"Nivel {level['id']}"
            level_name = level['name']
            
            # Estado del nivel (completado, bloqueado)
            if not level['unlocked']:
                status_text = "BLOQUEADO"
                status_color = self.colors['locked_gray']
            elif level['completed']:
                status_text = "COMPLETADO"
                status_color = self.colors['completed_green']
            else:
                status_text = "DISPONIBLE"
                status_color = self.colors['ice_blue']
            
            # Renderizar textos
            num_surface = self.font_small.render(level_num, True, self.colors['text_gold'])
            name_surface = self.font_text.render(level_name, True, self.colors['text_white'])
            status_surface = self.font_small.render(status_text, True, status_color)
            
            # Posiciones de texto
            text_x = x + 50
            self.screen.blit(num_surface, (text_x, y + 20))
            self.screen.blit(name_surface, (text_x, y + 40))
            self.screen.blit(status_surface, (text_x, y + 60))
            
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
            status_text = "¡Nivel listo para jugar! Enfréntate al reto y ayuda a restaurar el planeta."
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
                    self.selected_level_index = i
                    self.selected_level = level
                    self.show_level_info = True
                    self.animate_selected = True
                    return True
                else:
                    self.show_message("Este nivel está bloqueado. ¡Completa los niveles anteriores primero!", "error")
                    return False
        
        # Comprobar clics en botones
        for key, button in self.buttons.items():
            if button['rect'].collidepoint(pos):
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
        
        # Clic en espacio vacío
        self.show_level_info = False
        self.animate_selected = False
        return False
    
    def handle_hover(self, pos):
        """Manejar hover sobre elementos"""
        # Reset hover state
        self.hover_level_index = -1
        
        # Check hover over levels
        for i, level in enumerate(self.levels):
            if 'rect' in level and level['rect'].collidepoint(pos):
                self.hover_level_index = i
                return
    
    # Añadir método para efectos de transición
    def start_transition(self, direction):
        """Inicia una transición (1 para fade in, -1 para fade out)"""
        self.transitioning = True
        self.transition_direction = direction
        self.transition_alpha = 0 if direction > 0 else 255
    
    def update_transition(self):
        """Actualiza el estado de la transición"""
        if self.transitioning:
            # Actualizar valor alpha
            self.transition_alpha += 15 * self.transition_direction
            
            # Comprobar si la transición ha terminado
            if self.transition_alpha >= 255 or self.transition_alpha <= 0:
                self.transitioning = False
                return True
        return False
    
    def draw_transition(self):
        """Dibuja el efecto de transición"""
        if self.transitioning or self.transition_alpha > 0:
            # Crear superficie para el efecto de fade
            fade_surface = pygame.Surface((self.screen_width, self.screen_height))
            fade_surface.fill((0, 0, 0))  # Negro
            fade_surface.set_alpha(self.transition_alpha)
            self.screen.blit(fade_surface, (0, 0))
    
    def run(self):
        """Bucle principal"""
        running = True
        result = None
        
        # Efecto inicial de fade in
        self.start_transition(1)  # Fade in
        
        while running:
            dt = self.clock.tick(60) / 1000.0  # Delta time en segundos
            self.animation_time += dt
            
            # Actualizar partículas
            self.update_particles()
            
            # Actualizar efecto de transición
            if self.transitioning and self.update_transition() and self.transition_direction < 0:
                # Si terminamos un fade out, salimos del bucle
                running = False
            
            # Procesar eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.start_transition(-1)  # Iniciar fade out
                    running = False
                    result = None
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.transitioning:
                    if event.button == 1:  # Clic izquierdo
                        action = self.handle_click(event.pos)
                        if action and isinstance(action, str):
                            if action == 'back_to_menu' or action.startswith('start_level_'):
                                # Iniciar transición de salida
                                self.start_transition(-1)
                                result = action
                elif event.type == pygame.MOUSEMOTION and not self.transitioning:
                    self.handle_hover(event.pos)
                elif event.type == pygame.KEYDOWN and not self.transitioning:
                    if event.key == pygame.K_ESCAPE:
                        self.start_transition(-1)
                        result = 'back_to_menu'
            
            # Dibujar pantalla
            self.draw_background()
            self.draw_title()
            self.draw_level_cards()
            if self.show_level_info:
                self.draw_level_details()
            self.draw_buttons()
            self.draw_message()
            
            # Dibujar efecto de transición encima de todo
            self.draw_transition()
            
            # Actualizar pantalla
            pygame.display.flip()
        
        return result

# Función para ejecutar la pantalla de forma independiente (pruebas)
if __name__ == "__main__":
    screen = LevelSelectScreen()
    result = screen.run()
    print(f"Result: {result}")