import pygame
import sys
import math
import json
import os, random

class HockeyMainScreen:
    def __init__(self):
        pygame.init()
        
        # Configuración de pantalla adaptativa
        info = pygame.display.Info()
        self.screen_width = min(1200, info.current_w - 100)
        self.screen_height = min(800, info.current_h - 100)
        
        # Detectar si es formato móvil (relación de aspecto vertical)
        self.is_mobile = self.screen_height > self.screen_width
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Hockey Ice Melting Down - Salva la Tierra")
        
        # Colores temáticos
        self.colors = {
            'bg_gradient_top': (25, 25, 80),      # Azul espacial oscuro
            'bg_gradient_bottom': (120, 50, 50),   # Rojo apocalíptico
            'ice_blue': (173, 216, 230),
            'critical_red': (220, 50, 50),
            'hope_green': (34, 139, 34),
            'warning_orange': (255, 140, 0),
            'text_white': (255, 255, 255),
            'text_gold': (255, 215, 0),
            'panel_dark': (20, 20, 40, 200),
            'button_active': (0, 100, 200),
            'button_hover': (0, 150, 255)
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
        
        # Estado del juego
        self.game_data = {
            'player_points': 1420,
            'planetary_progress': {
                'oceanos_limpiados': 12,
                'ozono_restaurado': 8,
                'aire_purificado': 6,
                'bosques_replantados': 15,
                'ciudades_enfriadas': 3
            },
            'levels_unlocked': 2,
            'current_level': 1
        }
        
        # Agentes enemigos
        self.enemy_agents = [
            {'name': 'SLICKWAVE', 'desc': 'Emperador del plástico\nInunda los océanos con desechos', 'unlocked': True, 'defeated': False},
            {'name': 'UVBLADE', 'desc': 'Destructor del ozono\nHa perforado el escudo celestial', 'unlocked': True, 'defeated': False},
            {'name': 'SMOGATRON', 'desc': 'Señor del smog\nAhoga las ciudades en niebla tóxica', 'unlocked': False, 'defeated': False},
            {'name': 'DEFORESTIX', 'desc': 'Talador de raíces\nDevora los pulmones del planeta', 'unlocked': False, 'defeated': False},
            {'name': 'HEATCORE', 'desc': 'Maestro del calor\nConvierte ciudades en hornos', 'unlocked': False, 'defeated': False}
        ]
        
        # Botones circulares
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        if self.is_mobile:
            # Disposición vertical para móvil
            self.buttons = {
                'play': {'pos': (center_x, center_y - 80), 'radius': 50, 'color': self.colors['hope_green']},
                'history': {'pos': (center_x - 80, center_y + 40), 'radius': 30, 'color': self.colors['ice_blue']},
                'player': {'pos': (center_x + 80, center_y + 40), 'radius': 30, 'color': self.colors['warning_orange']},
                'settings': {'pos': (30, 30), 'radius': 20, 'color': self.colors['ice_blue']},
                'help': {'pos': (self.screen_width - 30, 30), 'radius': 20, 'color': self.colors['ice_blue']}
            }
        else:
            # Disposición para PC
            self.buttons = {
                'play': {'pos': (center_x, center_y), 'radius': 60, 'color': self.colors['hope_green']},
                'history': {'pos': (center_x - 120, center_y), 'radius': 35, 'color': self.colors['ice_blue']},
                'player': {'pos': (center_x + 120, center_y), 'radius': 35, 'color': self.colors['warning_orange']},
                'background': {'pos': (center_x, center_y + 80), 'radius': 30, 'color': self.colors['button_active']},
                'settings': {'pos': (40, 40), 'radius': 25, 'color': self.colors['ice_blue']},
                'help': {'pos': (self.screen_width - 40, 40), 'radius': 25, 'color': self.colors['ice_blue']}
            }
        
        # Animaciones
        self.animation_time = 0
        self.particles = []
        self.create_particles()
        
        # Paneles
        self.show_gaia_panel = False
        self.show_progress_panel = True
        
        self.clock = pygame.time.Clock()
        
    def create_particles(self):
        """Crear partículas ambientales para el efecto atmosférico"""
        for _ in range(20):
            self.particles.append({
                'x': pygame.math.Vector2(
                    random.randint(0, self.screen_width),
                    random.randint(0, self.screen_height)
                ),
                'vel': pygame.math.Vector2(
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5)
                ),
                'size': random.randint(1, 3),
                'color': random.choice([
                    self.colors['ice_blue'],
                    self.colors['warning_orange'],
                    (100, 100, 150)
                ])
            })
    
    def draw_gradient_background(self):
        """Dibujar fondo con gradiente dramático"""
        for y in range(self.screen_height):
            ratio = y / self.screen_height
            color = [
                int(self.colors['bg_gradient_top'][i] * (1 - ratio) + 
                    self.colors['bg_gradient_bottom'][i] * ratio)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))
    
    def draw_particles(self):
        """Dibujar y animar partículas atmosféricas"""
        for particle in self.particles:
            particle['x'] += particle['vel']
            
            # Wraparound
            if particle['x'].x < 0:
                particle['x'].x = self.screen_width
            elif particle['x'].x > self.screen_width:
                particle['x'].x = 0
            if particle['x'].y < 0:
                particle['x'].y = self.screen_height
            elif particle['x'].y > self.screen_height:
                particle['x'].y = 0
            
            # Dibujar partícula con efecto de brillo
            alpha = int(128 + 127 * math.sin(self.animation_time * 2 + particle['x'].x * 0.01))
            color = (*particle['color'], alpha)
            
            # Crear superficie temporal para transparencia
            temp_surface = pygame.Surface((particle['size'] * 2, particle['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, color, (particle['size'], particle['size']), particle['size'])
            self.screen.blit(temp_surface, (particle['x'].x - particle['size'], particle['x'].y - particle['size']))
    
    def draw_title(self):
        """Dibujar título principal del juego"""
        # Título principal con efecto de brillo
        title_text = "HOCKEY ICE"
        subtitle_text = "MELTING DOWN"
        
        # Posición adaptativa
        title_y = 80 if not self.is_mobile else 50
        
        # Efecto de brillo en el título
        glow_offset = int(5 * math.sin(self.animation_time * 3))
        
        # Sombra del título
        title_shadow = self.font_title.render(title_text, True, (0, 0, 0))
        subtitle_shadow = self.font_subtitle.render(subtitle_text, True, (0, 0, 0))
        
        title_rect = title_shadow.get_rect(center=(self.screen_width // 2 + 2, title_y + 2))
        subtitle_rect = subtitle_shadow.get_rect(center=(self.screen_width // 2 + 2, title_y + 50 + 2))
        
        self.screen.blit(title_shadow, title_rect)
        self.screen.blit(subtitle_shadow, subtitle_rect)
        
        # Título principal
        title_surface = self.font_title.render(title_text, True, self.colors['ice_blue'])
        subtitle_surface = self.font_subtitle.render(subtitle_text, True, self.colors['critical_red'])
        
        title_rect = title_surface.get_rect(center=(self.screen_width // 2, title_y))
        subtitle_rect = subtitle_surface.get_rect(center=(self.screen_width // 2, title_y + 50))
        
        self.screen.blit(title_surface, title_rect)
        self.screen.blit(subtitle_surface, subtitle_rect)
        
        # Estado planetario crítico
        status_text = "ESTADO PLANETARIO: CRÍTICO"
        status_surface = self.font_text.render(status_text, True, self.colors['critical_red'])
        status_rect = status_surface.get_rect(center=(self.screen_width // 2, title_y + 90))
        self.screen.blit(status_surface, status_rect)
    
    def draw_circular_button(self, button_key, icon_text, hover_text=""):
        """Dibujar botón circular con efectos"""
        button = self.buttons[button_key]
        pos = button['pos']
        radius = button['radius']
        color = button['color']
        
        # Efecto de pulsación
        pulse = int(3 * math.sin(self.animation_time * 4))
        current_radius = radius + pulse
        
        # Detectar hover
        mouse_pos = pygame.mouse.get_pos()
        distance = math.sqrt((mouse_pos[0] - pos[0])**2 + (mouse_pos[1] - pos[1])**2)
        is_hover = distance <= radius
        
        if is_hover:
            current_radius += 5
            color = self.colors['button_hover']
        
        # Dibujar círculo con borde
        pygame.draw.circle(self.screen, color, pos, current_radius)
        pygame.draw.circle(self.screen, self.colors['text_white'], pos, current_radius, 3)
        
        # Dibujar icono/texto
        if button_key == 'play':
            # Triángulo de play
            triangle_points = [
                (pos[0] - 15, pos[1] - 20),
                (pos[0] - 15, pos[1] + 20),
                (pos[0] + 20, pos[1])
            ]
            pygame.draw.polygon(self.screen, self.colors['text_white'], triangle_points)
        else:
            # Texto del icono
            icon_surface = self.font_text.render(icon_text, True, self.colors['text_white'])
            icon_rect = icon_surface.get_rect(center=pos)
            self.screen.blit(icon_surface, icon_rect)
        
        # Mostrar texto de hover
        if is_hover and hover_text:
            hover_surface = self.font_small.render(hover_text, True, self.colors['text_white'])
            hover_rect = hover_surface.get_rect(center=(pos[0], pos[1] + radius + 20))
            self.screen.blit(hover_surface, hover_rect)
        
        return is_hover
    
    def draw_gaia_panel(self):
        """Dibujar panel de información de GAIA"""
        if not self.show_gaia_panel:
            return
        
        panel_width = 300 if not self.is_mobile else self.screen_width - 40
        panel_height = 250 if not self.is_mobile else 200
        panel_x = 20 if not self.is_mobile else 20
        panel_y = self.screen_height // 2 - 50 if not self.is_mobile else self.screen_height // 2 + 50
        
        # Fondo del panel con transparencia
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill(self.colors['panel_dark'])
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Borde del panel
        pygame.draw.rect(self.screen, self.colors['hope_green'], 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Título del panel
        title = self.font_subtitle.render("ARCHIVO GAIA", True, self.colors['hope_green'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        subtitle = self.font_text.render("GAIA CORE - SISTEMA DE RESTAURACIÓN", True, self.colors['text_white'])
        self.screen.blit(subtitle, (panel_x + 10, panel_y + 35))
        
        # Descripción de la situación
        situation_text = [
            "La Tierra agoniza. Los polos se derriten,",
            "los bosques desaparecen y las ciudades",
            "arden bajo olas de calor implacables.",
            "",
            "Cinco agentes corruptos de EcoNull",
            "han tomado control del clima:",
            "",
            "🌊 SLICKWAVE - Emperador del plástico",
            "☀️ UVBLADE - Destructor del ozono"
        ]
        
        y_offset = 60
        for line in situation_text:
            if line.strip():
                color = self.colors['text_gold'] if line.startswith(('🌊', '☀️')) else self.colors['text_white']
                text_surface = self.font_small.render(line, True, color)
                self.screen.blit(text_surface, (panel_x + 10, panel_y + y_offset))
            y_offset += 18
    
    def draw_progress_panel(self):
        """Dibujar panel de progreso planetario"""
        if not self.show_progress_panel:
            return
        
        panel_width = 250 if not self.is_mobile else self.screen_width - 40
        panel_height = 180 if not self.is_mobile else 150
        panel_x = self.screen_width - panel_width - 20 if not self.is_mobile else 20
        panel_y = self.screen_height // 2 - 50 if not self.is_mobile else 20
        
        # Fondo del panel
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill(self.colors['panel_dark'])
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Borde del panel
        pygame.draw.rect(self.screen, self.colors['ice_blue'], 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Título
        title = self.font_subtitle.render("PROGRESO PLANETARIO", True, self.colors['ice_blue'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Barras de progreso
        progress_items = [
            ("Océanos limpiados:", self.game_data['planetary_progress']['oceanos_limpiados'], self.colors['ice_blue']),
            ("Ozono restaurado:", self.game_data['planetary_progress']['ozono_restaurado'], self.colors['warning_orange']),
            ("Aire purificado:", self.game_data['planetary_progress']['aire_purificado'], self.colors['hope_green']),
            ("Bosques replantados:", self.game_data['planetary_progress']['bosques_replantados'], self.colors['hope_green']),
            ("Ciudades enfriadas:", self.game_data['planetary_progress']['ciudades_enfriadas'], self.colors['ice_blue'])
        ]
        
        y_offset = 40
        for label, value, color in progress_items:
            # Etiqueta
            label_surface = self.font_small.render(f"{label}", True, self.colors['text_white'])
            self.screen.blit(label_surface, (panel_x + 10, panel_y + y_offset))
            
            # Valor porcentual
            percent_surface = self.font_small.render(f"{value}%", True, color)
            self.screen.blit(percent_surface, (panel_x + panel_width - 40, panel_y + y_offset))
            
            # Barra de progreso
            bar_width = panel_width - 80
            bar_height = 8
            bar_x = panel_x + 10
            bar_y = panel_y + y_offset + 15
            
            # Fondo de la barra
            pygame.draw.rect(self.screen, (50, 50, 50), 
                           (bar_x, bar_y, bar_width, bar_height))
            
            # Progreso
            progress_width = int((value / 100) * bar_width)
            pygame.draw.rect(self.screen, color, 
                           (bar_x, bar_y, progress_width, bar_height))
            
            y_offset += 25
        
        # Puntos GAIA
        points_text = f"Puntos Gaia: {self.game_data['player_points']}"
        points_surface = self.font_text.render(points_text, True, self.colors['text_gold'])
        self.screen.blit(points_surface, (panel_x + 10, panel_y + panel_height - 25))
    
    def draw_climate_warning(self):
        """Dibujar mensaje de advertencia climática en la parte inferior"""
        warning_texts = [
            "Los glaciares pierden 280 mil millones de toneladas anuales",
            "La temperatura global ha aumentado 1.5°C desde 1880",
            "El nivel del mar sube 3.3mm cada año",
            "Quedan menos de 10 años para actuar"
        ]
        
        # Alternar mensajes cada 3 segundos
        message_index = int(self.animation_time / 3) % len(warning_texts)
        current_message = warning_texts[message_index]
        
        # Fondo para el mensaje
        warning_surface = self.font_text.render(current_message, True, self.colors['text_white'])
        warning_rect = warning_surface.get_rect(center=(self.screen_width // 2, self.screen_height - 30))
        
        # Fondo con borde naranja
        bg_rect = pygame.Rect(warning_rect.x - 10, warning_rect.y - 5, 
                             warning_rect.width + 20, warning_rect.height + 10)
        pygame.draw.rect(self.screen, self.colors['warning_orange'], bg_rect)
        pygame.draw.rect(self.screen, self.colors['text_white'], bg_rect, 2)
        
        self.screen.blit(warning_surface, warning_rect)
    
    def handle_click(self, pos):
        """Manejar clics en botones"""
        for button_key, button in self.buttons.items():
            distance = math.sqrt((pos[0] - button['pos'][0])**2 + (pos[1] - button['pos'][1])**2)
            if distance <= button['radius']:
                if button_key == 'play':
                    print("Iniciando juego...")
                    return 'start_game'
                elif button_key == 'history':
                    self.show_gaia_panel = not self.show_gaia_panel
                elif button_key == 'player':
                    print("Abriendo perfil de jugador...")
                    return 'player_profile'
                elif button_key == 'settings':
                    print("Abriendo configuración...")
                    return 'settings'
                elif button_key == 'help':
                    print("Abriendo ayuda...")
                    return 'help'
                elif button_key == 'background':
                    print("Cambiar fondo temático...")
                    return 'background'
        return None
    
    def run(self):
        """Bucle principal del menú"""
        running = True
        
        while running:
            dt = self.clock.tick(60) / 1000.0
            self.animation_time += dt
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Clic izquierdo
                        action = self.handle_click(event.pos)
                        if action == 'start_game':
                            running = False  # Salir para iniciar el juego
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        action = self.handle_click(self.buttons['play']['pos'])
                        if action == 'start_game':
                            running = False
            
            # Dibujar todo
            self.draw_gradient_background()
            self.draw_particles()
            self.draw_title()
            
            # Dibujar botones con efectos hover
            self.draw_circular_button('play', '▶', "Comenzar Misión")
            self.draw_circular_button('history', '📜', "Historial de Partidas")
            self.draw_circular_button('player', '👤', "Personalizar Jugador")
            self.draw_circular_button('settings', '⚙', "Configuración")
            self.draw_circular_button('help', '?', "Ayuda")
            
            if not self.is_mobile:
                self.draw_circular_button('background', '🎨', "Fondo Temático")
            
            # Dibujar paneles informativos
            self.draw_gaia_panel()
            self.draw_progress_panel()
            self.draw_climate_warning()
            
            pygame.display.flip()
        
        pygame.quit()
        return "start_game"

# Ejecutar el menú principal
if __name__ == "__main__":
    main_screen = HockeyMainScreen()
    result = main_screen.run()
    print(f"Resultado: {result}")