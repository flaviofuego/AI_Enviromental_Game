import pygame
import sys
import math
import json
import os, random
from datetime import datetime
import time

from ..config.save_system import GameSaveSystem
from ..components.Button import Button
from ..components.PopUp import PopUp, create_help_popup

class HockeyMainScreen:
    def __init__(self, screen=None, save_system=None):
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
            pygame.display.set_caption("Hockey Is Melting Down - Salva la Tierra")
        else:
            # Usar la ventana existente
            self.screen = screen
            self.screen_width = screen.get_width()
            self.screen_height = screen.get_height()
            self.is_mobile = self.screen_height > self.screen_width
        
        # Cargar imagen de fondo
        self.background_image = pygame.image.load('game/assets/background.png')
        self.background_opacity = 200  # 0-255, donde 255 es completamente opaco
        
        # Inicializar sistema de guardado
        self.save_system = save_system if save_system else GameSaveSystem()
        
        # Estado de interfaz
        self.current_screen = "main"  # main, profiles, create_profile, settings
        self.profiles = []
        self.selected_profile_index = -1
        self.input_text = ""
        self.input_active = False
        self.error_message = ""
        self.success_message = ""
        self.message_time = 0
        
        # Colores temáticos
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
            'purple': (128, 0, 128),
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
        
        # Estado del juego - Ahora se cargará desde el perfil
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
                'play': {'pos': (center_x, center_y - 80), 'scale': (100, 100), 'tex_hover': 'Jugar'},
                'history': {'pos': (center_x - 80, center_y + 40), 'scale': (30, 30), 'tex_hover': 'Historial'},
                'player': {'pos': (center_x + 80, center_y + 40), 'scale': (30, 30), 'tex_hover': 'Jugador'},
                'settings': {'pos': (30, 30), 'scale': (30, 30), 'tex_hover': 'Configuración'},
                'help': {'pos': (self.screen_width - 30, 30), 'radius': 30, 'color': self.colors['ice_blue']}
            }
        else:
            # Disposición para PC
            self.buttons = {
                'play': {'pos': (center_x, center_y), 'scale': (300, 300), 'tex_hover': 'Jugar'},
                'history': {'pos': (center_x - 220, center_y), 'scale': (120, 120), 'tex_hover': 'Historial'},
                'player': {'pos': (center_x + 220, center_y), 'scale': (120, 120), 'tex_hover': 'Jugador'},
                #'background': {'pos': (center_x, center_y + 120), 'radius': 30, 'color': self.colors['button_active']},
                'settings': {'pos': (40, 40), 'scale': (80, 80), 'tex_hover': 'Configuración'},
                'help': {'pos': (self.screen_width - 40, 40), 'scale': (80, 80), 'tex_hover': 'Ayuda'}
            }
        
        # Animaciones
        self.animation_time = 0
        self.particles = []
        self.create_particles()
        
        # Nuevos efectos animados para el fondo
        self.pollution_particles = []
        self.heat_waves = []
        self.acid_rain = []
        self.melting_ice = []
        self.hope_sparkles = []
        self.aurora_strips = []
        self.create_environmental_effects()
        
        # Paneles
        self.show_gaia_panel = False
        self.show_progress_panel = True
        self.help_popup = None
        
        # Cargar perfiles al iniciar
        self.load_profiles()
        
        self.clock = pygame.time.Clock()
    
    def load_profiles(self):
        """Carga la lista de perfiles disponibles y el último perfil usado"""
        self.profiles = self.save_system.get_all_profiles()
        
        # Si hay perfiles disponibles, cargar el primero (el más reciente)
        if self.profiles:
            self.selected_profile_index = 0
            # Cargar automáticamente el último perfil usado
            profile_id = self.profiles[0]["profile_id"]  # El primer perfil es el más reciente
            self.load_selected_profile()
            
            # No mostrar mensaje de bienvenida al inicio
            self.success_message = ""
            self.message_time = 0
    
    def load_selected_profile(self):
        """Carga el perfil seleccionado"""
        if 0 <= self.selected_profile_index < len(self.profiles):
            profile_id = self.profiles[self.selected_profile_index]["profile_id"]
            profile = self.save_system.load_profile(profile_id)
            
            if profile:
                # Actualizar datos del juego con el perfil
                self.game_data = {
                    'player_points': profile["total_points"],
                    'planetary_progress': profile["planetary_progress"],
                    'levels_unlocked': profile["levels"]["unlocked"],
                    'current_level': profile["levels"]["current"]
                }
                
                # Actualizar estado de enemigos
                for enemy in self.enemy_agents:
                    enemy["defeated"] = enemy["name"] in profile["enemies_defeated"]
                    # Desbloquear enemigos según nivel
                    if self.game_data['levels_unlocked'] >= self.enemy_agents.index(enemy) + 1:
                        enemy["unlocked"] = True
                
                self.success_message = f"¡Bienvenido de vuelta, {profile['player_name']}!"
                self.message_time = time.time()
                
                # Volver al menú principal
                self.current_screen = "main"
                return True
            else:
                self.error_message = "Error al cargar el perfil."
                self.message_time = time.time()
                return False
        return False
    
    def create_new_profile(self):
        """Crea un nuevo perfil con el nombre ingresado"""
        if not self.input_text.strip():
            self.error_message = "Por favor ingresa un nombre de jugador."
            self.message_time = time.time()
            return False
            
        name = self.input_text.strip()
        profile_id = self.save_system.create_profile(name)
        
        if profile_id:
            # Reiniciar datos del juego para el nuevo perfil
            profile = self.save_system.get_current_profile_data()
            self.game_data = {
                'player_points': 0,
                'planetary_progress': profile["planetary_progress"],
                'levels_unlocked': 1,
                'current_level': 1
            }
            
            # Restablecer estado de enemigos
            for enemy in self.enemy_agents:
                enemy["defeated"] = False
                enemy["unlocked"] = enemy == self.enemy_agents[0]  # Solo el primero desbloqueado
            
            self.success_message = f"¡Perfil creado! Bienvenido, {name}."
            self.message_time = time.time()
            
            # Actualizar lista de perfiles
            self.load_profiles()
            
            # Volver al menú principal
            self.current_screen = "main"
            return True
        else:
            self.error_message = "Error al crear el perfil."
            self.message_time = time.time()
            return False
    
    def delete_selected_profile(self):
        """Elimina el perfil seleccionado"""
        if 0 <= self.selected_profile_index < len(self.profiles):
            profile_id = self.profiles[self.selected_profile_index]["profile_id"]
            if self.save_system.delete_profile(profile_id):
                self.success_message = "Perfil eliminado."
                self.message_time = time.time()
                
                # Actualizar lista
                self.load_profiles()
                
                # Resetear selección
                self.selected_profile_index = 0 if self.profiles else -1
                return True
            else:
                self.error_message = "Error al eliminar el perfil."
                self.message_time = time.time()
        return False
    
    def save_game_progress(self):
        """Guarda el progreso actual del juego"""
        if not self.save_system.current_profile:
            self.error_message = "No hay un perfil activo para guardar."
            self.message_time = time.time()
            return False
            
        # Preparar datos a guardar
        game_data = {
            "points": self.game_data['player_points'],
            "planetary_progress": self.game_data['planetary_progress'],
            "level_completed": self.game_data['current_level'] - 1,  # El nivel actual ya está completado
            "stats": {
                "games_played": 1,
                "time_played": 60  # Ejemplo: 60 segundos de juego
            }
        }
        
        # Si hay enemigos derrotados, añadirlos
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
    
    def create_environmental_effects(self):
        """Crear efectos ambientales animados"""
        # Partículas de contaminación (humo negro/gris)
        for _ in range(15):
            self.pollution_particles.append({
                'x': random.randint(0, self.screen_width),
                'y': random.randint(0, self.screen_height),
                'vel_x': random.uniform(-0.3, 0.3),
                'vel_y': random.uniform(-0.8, -0.2),
                'size': random.randint(3, 8),
                'alpha': random.randint(30, 80),
                'life': random.randint(200, 400)
            })
        
        # Ondas de calor
        for _ in range(8):
            self.heat_waves.append({
                'y': random.randint(100, self.screen_height - 100),
                'amplitude': random.randint(10, 30),
                'frequency': random.uniform(0.02, 0.05),
                'speed': random.uniform(1, 3),
                'offset': random.uniform(0, 6.28)
            })
        
        # Lluvia ácida
        for _ in range(25):
            self.acid_rain.append({
                'x': random.randint(-50, self.screen_width + 50),
                'y': random.randint(-100, -10),
                'vel_y': random.uniform(2, 5),
                'vel_x': random.uniform(-0.5, 0.5),
                'length': random.randint(5, 15)
            })
        
        # Hielo derritiéndose
        for _ in range(6):
            self.melting_ice.append({
                'x': random.randint(50, self.screen_width - 50),
                'y': random.randint(50, 200),
                'drops': [],
                'last_drop': 0
            })
        
        # Chispas de esperanza (verde)
        for _ in range(10):
            self.hope_sparkles.append({
                'x': random.randint(0, self.screen_width),
                'y': random.randint(0, self.screen_height),
                'vel_x': random.uniform(-0.5, 0.5),
                'vel_y': random.uniform(-0.5, 0.5),
                'size': random.randint(1, 3),
                'pulse': random.uniform(0, 6.28),
                'color_intensity': random.randint(100, 255)
            })
        
        # Tiras de aurora dañada
        for _ in range(5):
            self.aurora_strips.append({
                'points': [(random.randint(0, self.screen_width), random.randint(0, 150)) for _ in range(6)],
                'color_shift': random.uniform(0, 6.28),
                'flicker_intensity': random.uniform(0.3, 0.8)
            })

    def update_environmental_effects(self, dt):
        """Actualizar todos los efectos ambientales"""
        # Actualizar partículas de contaminación
        for particle in self.pollution_particles:
            particle['x'] += particle['vel_x']
            particle['y'] += particle['vel_y']
            particle['life'] -= 1
            
            if particle['life'] <= 0 or particle['y'] < -20:
                particle['x'] = random.randint(0, self.screen_width)
                particle['y'] = self.screen_height + 20
                particle['life'] = random.randint(200, 400)
        
        # Actualizar ondas de calor
        for wave in self.heat_waves:
            wave['offset'] += wave['speed'] * dt
        
        # Actualizar lluvia ácida
        for drop in self.acid_rain:
            drop['x'] += drop['vel_x']
            drop['y'] += drop['vel_y']
            
            if drop['y'] > self.screen_height + 10:
                drop['x'] = random.randint(-50, self.screen_width + 50)
                drop['y'] = random.randint(-100, -10)
        
        # Actualizar hielo derritiéndose
        for ice in self.melting_ice:
            ice['last_drop'] += dt
            if ice['last_drop'] > random.uniform(0.5, 2.0):
                ice['drops'].append({
                    'x': ice['x'] + random.randint(-5, 5),
                    'y': ice['y'],
                    'vel_y': random.uniform(1, 3)
                })
                ice['last_drop'] = 0
            
            # Actualizar gotas de hielo
            for drop in ice['drops'][:]:
                drop['y'] += drop['vel_y']
                if drop['y'] > self.screen_height:
                    ice['drops'].remove(drop)
        
        # Actualizar chispas de esperanza
        for sparkle in self.hope_sparkles:
            sparkle['x'] += sparkle['vel_x']
            sparkle['y'] += sparkle['vel_y']
            sparkle['pulse'] += dt * 3
            
            # Wraparound
            if sparkle['x'] < 0:
                sparkle['x'] = self.screen_width
            elif sparkle['x'] > self.screen_width:
                sparkle['x'] = 0
            if sparkle['y'] < 0:
                sparkle['y'] = self.screen_height
            elif sparkle['y'] > self.screen_height:
                sparkle['y'] = 0

    def draw_animated_background(self):
        """Dibujar fondo animado con efectos ambientales"""
        
        # Dibujar el gradiente base
        #self.draw_gradient_background()
        
        # Dibujar la imagen de fondo con transparencia
        if self.background_image:
            # Crear una copia de la imagen con la opacidad deseada
            temp_image = self.background_image.copy()
            temp_image.set_alpha(self.background_opacity)
            self.screen.blit(temp_image, (0, 0))
        
        # Continuar con los efectos animados
        self.draw_damaged_aurora()
        self.draw_heat_waves()
        self.draw_acid_rain()
        self.draw_pollution_particles()
        self.draw_melting_ice()
        self.draw_hope_sparkles()
        self.draw_particles()

    def draw_damaged_aurora(self):
        """Dibujar aurora boreal dañada que parpadea"""
        for aurora in self.aurora_strips:
            # Calcular intensidad con parpadeo
            flicker = math.sin(self.animation_time * 2 + aurora['color_shift']) * aurora['flicker_intensity']
            base_alpha = int(30 + 25 * flicker)
            
            if base_alpha > 10:
                # Colores de aurora dañada (rojizos y verdes tóxicos)
                colors = [
                    (255, 100, 100, base_alpha),  # Rojo tóxico
                    (100, 255, 100, base_alpha),  # Verde ácido
                    (150, 100, 255, base_alpha)   # Púrpura artificial
                ]
                
                for i, color in enumerate(colors):
                    # Crear superficie para transparencia
                    aurora_surface = pygame.Surface((self.screen_width, 80), pygame.SRCALPHA)
                    
                    # Dibujar líneas onduladas
                    points = []
                    for j in range(len(aurora['points'])):
                        x = aurora['points'][j][0] + 20 * math.sin(self.animation_time + j * 0.5)
                        y = aurora['points'][j][1] + i * 15 + 10 * math.sin(self.animation_time * 1.5 + j)
                        points.append((x, y))
                    
                    if len(points) > 2:
                        pygame.draw.lines(aurora_surface, color, False, points, 3)
                    
                    self.screen.blit(aurora_surface, (0, 0))

    def draw_heat_waves(self):
        """Dibujar ondas de calor distorsionadas"""
        for wave in self.heat_waves:
            points = []
            for x in range(0, self.screen_width, 10):
                y_offset = wave['amplitude'] * math.sin(x * wave['frequency'] + wave['offset'])
                points.append((x, wave['y'] + y_offset))
            
            if len(points) > 1:
                # Dibujar ondas semitransparentes
                wave_surface = pygame.Surface((self.screen_width, 4), pygame.SRCALPHA)
                for i in range(len(points) - 1):
                    alpha = int(40 + 20 * math.sin(self.animation_time * 2 + points[i][0] * 0.01))
                    color = (255, 150, 50, alpha)  # Naranja cálido
                    pygame.draw.line(wave_surface, color, points[i], points[i + 1], 2)
                
                self.screen.blit(wave_surface, (0, 0))

    def draw_acid_rain(self):
        """Dibujar lluvia ácida"""
        for drop in self.acid_rain:
            # Color amarillo-verde tóxico
            color = (200, 255, 100)
            start_pos = (int(drop['x']), int(drop['y']))
            end_pos = (int(drop['x'] + drop['vel_x'] * 2), int(drop['y'] + drop['length']))
            
            if 0 <= start_pos[0] <= self.screen_width and 0 <= start_pos[1] <= self.screen_height:
                pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

    def draw_pollution_particles(self):
        """Dibujar partículas de contaminación"""
        for particle in self.pollution_particles:
            if particle['life'] > 0:
                # Crear partícula de humo con transparencia
                particle_surface = pygame.Surface((particle['size'] * 2, particle['size'] * 2), pygame.SRCALPHA)
                
                # Color gris/negro con transparencia
                alpha = min(particle['alpha'], particle['life'] // 2)
                color = (60, 60, 60, alpha)
                
                pygame.draw.circle(particle_surface, color, 
                                 (particle['size'], particle['size']), particle['size'])
                
                self.screen.blit(particle_surface, 
                               (particle['x'] - particle['size'], particle['y'] - particle['size']))

    def draw_melting_ice(self):
        """Dibujar hielo derritiéndose"""
        for ice in self.melting_ice:
            # Dibujar bloque de hielo (cada vez más pequeño)
            ice_size = 20 + 10 * math.sin(self.animation_time * 0.5)
            ice_color = (200, 230, 255, 150)
            
            ice_surface = pygame.Surface((ice_size, ice_size), pygame.SRCALPHA)
            pygame.draw.rect(ice_surface, ice_color, (0, 0, ice_size, ice_size))
            self.screen.blit(ice_surface, (ice['x'] - ice_size//2, ice['y'] - ice_size//2))
            
            # Dibujar gotas de agua cayendo
            for drop in ice['drops']:
                pygame.draw.circle(self.screen, (100, 150, 255), 
                                 (int(drop['x']), int(drop['y'])), 2)

    def draw_hope_sparkles(self):
        """Dibujar chispas de esperanza (elementos verdes de vida)"""
        for sparkle in self.hope_sparkles:
            # Pulsación de brillo
            pulse_factor = (math.sin(sparkle['pulse']) + 1) * 0.5
            alpha = int(sparkle['color_intensity'] * pulse_factor)
            
            if alpha > 20:
                # Crear estrella de esperanza
                sparkle_surface = pygame.Surface((sparkle['size'] * 4, sparkle['size'] * 4), pygame.SRCALPHA)
                
                # Verde esperanzador
                color = (50, 255, 100, alpha)
                center = (sparkle['size'] * 2, sparkle['size'] * 2)
                
                # Dibujar como pequeña estrella
                points = []
                for i in range(8):
                    angle = i * math.pi / 4
                    if i % 2 == 0:
                        radius = sparkle['size'] + 2
                    else:
                        radius = sparkle['size'] // 2
                    
                    x = center[0] + radius * math.cos(angle + sparkle['pulse'])
                    y = center[1] + radius * math.sin(angle + sparkle['pulse'])
                    points.append((x, y))
                
                if len(points) > 2:
                    pygame.draw.polygon(sparkle_surface, color, points)
                
                self.screen.blit(sparkle_surface, 
                               (sparkle['x'] - sparkle['size'] * 2, sparkle['y'] - sparkle['size'] * 2))

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
            
            # Dibujar partícula with efecto de brillo
            alpha = int(128 + 127 * math.sin(self.animation_time * 2 + particle['x'].x * 0.01))
            color = (*particle['color'], alpha)
            
            # Crear superficie temporal para transparencia
            temp_surface = pygame.Surface((particle['size'] * 2, particle['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, color, (particle['size'], particle['size']), particle['size'])
            self.screen.blit(temp_surface, (particle['x'].x - particle['size'], particle['x'].y - particle['size']))
    
    def draw_title(self):
        """Dibujar título principal del juego"""
        # Título principal con efecto de brillo
        title_text = "HOCKEY IS MELTING DOWN"
        subtitle_text = "Desafía a EcoNull y salva la Tierra"
        
        # Posición adaptativa
        title_y = 100 if not self.is_mobile else 70
        
        # Calcular posición del subtítulo
        subtitle_y = title_y + 40

        # Efecto de brillo/titilación para el título
        glow_intensity = abs(math.sin(self.animation_time * 4)) # Titilación más rápida
        glow_surfaces = []
        
        # Crear capas de brillo con intensidad variable
        for offset in range(3, 0, -1):
            alpha = int(50 * glow_intensity) # La transparencia varía con la intensidad
            glow_color = (100 + offset * 50, 150 + offset * 30, 255, alpha)
            glow_surface = self.font_title.render(title_text, True, glow_color)
            glow_rect = glow_surface.get_rect(center=(self.screen_width // 2, title_y))
            glow_surfaces.append((glow_surface, glow_rect))

        # Dibujar las capas de brillo
        for glow_surface, glow_rect in glow_surfaces:
            self.screen.blit(glow_surface, glow_rect)
        
        # Sombra del título
        title_shadow = self.font_title.render(title_text, True, (0, 0, 0))
        subtitle_shadow = self.font_subtitle.render(subtitle_text, True, (0, 0, 0))
        
        title_rect = title_shadow.get_rect(center=(self.screen_width // 2 + 2, title_y + 2))
        subtitle_rect = subtitle_shadow.get_rect(center=(self.screen_width // 2 + 2, subtitle_y + 2))

        self.screen.blit(title_shadow, title_rect)
        self.screen.blit(subtitle_shadow, subtitle_rect)
        
        # Color principal del título que también titila ligeramente
        title_color = (
            min(255, int(173 + 80 * glow_intensity)),  # Componente R
            min(255, int(216 + 40 * glow_intensity)),  # Componente G
            min(255, int(230 + 25 * glow_intensity))   # Componente B
        )
        
        # Título principal
        title_surface = self.font_title.render(title_text, True, title_color)
        subtitle_surface = self.font_subtitle.render(subtitle_text, True, self.colors['critical_red'])
        
        title_rect = title_surface.get_rect(center=(self.screen_width // 2, title_y))
        subtitle_rect = subtitle_surface.get_rect(center=(self.screen_width // 2, subtitle_y))

        self.screen.blit(title_surface, title_rect)
        self.screen.blit(subtitle_surface, subtitle_rect)
        
        # Estado planetario crítico
        status_prefix = "ESTADO PLANETARIO: "
        status_critical = "CRÍTICO"

        # Renderizar "ESTADO PLANETARIO: "
        prefix_surface = self.font_text.render(status_prefix, True, self.colors['critical_red'])
        prefix_rect = prefix_surface.get_rect(center=(self.screen_width // 2 - 40, title_y + 90))
        
        # Renderizar "CRÍTICO" con efecto de brillo/fuego
        glow_intensity = abs(math.sin(self.animation_time * 4))
        critical_color = (
            255,  # R - Rojo máximo
            int(100 + 155 * glow_intensity),  # G - Amarillo variable
            0    # B - Sin azul para efecto fuego
        )
        
        critical_surface = self.font_text.render(status_critical, True, critical_color)
        critical_rect = critical_surface.get_rect(midleft=(prefix_rect.right, prefix_rect.centery))
        
        # Dibujar ambas partes
        self.screen.blit(prefix_surface, prefix_rect)
        self.screen.blit(critical_surface, critical_rect)
    
    def draw_circular_button(self, button_object: Button):
        """Dibujar botón circular con efectos"""
        button_object.draw(self.screen)
        is_hovered = button_object.is_hovered()

        self.buttons[button_object.name]['object'] = button_object

        return is_hovered

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
        panel_height = 210 if not self.is_mobile else 150
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
        self.screen.blit(points_surface, (panel_x + panel_width - points_surface.get_width() - 10, panel_y + panel_height - 25))
    
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
        
        warning_font = pygame.font.Font(None, 22)
        # Fondo para el mensaje
        warning_surface = warning_font.render(current_message, True, self.colors['text_white'])
        warning_rect = warning_surface.get_rect(center=(self.screen_width // 2, self.screen_height - 50))
        
        bg_rect = pygame.Rect(warning_rect.x - 15, warning_rect.y - 10,
                            warning_rect.width + 30, warning_rect.height + 20)
        pygame.draw.rect(self.screen, self.colors['warning_orange'], bg_rect)
        pygame.draw.rect(self.screen, self.colors['text_white'], bg_rect, 3)
        
        self.screen.blit(warning_surface, warning_rect)
    
    def handle_click(self, pos):
        """Manejar clics en botones del menú principal"""
        for button_key, button in self.buttons.items():
            if button['object'].is_clicked(pos):
                if button_key == 'play':
                    print("Iniciando selección de niveles...")
                    
                    # Verificar si hay un perfil activo
                    if not self.save_system.current_profile:
                        self.current_screen = "profiles"
                        self.error_message = "Selecciona o crea un perfil para jugar"
                        self.message_time = time.time()
                        return None
                    
                    return 'level_select'  # Cambiado de 'start_game' a 'level_select'
                elif button_key == 'history':
                    self.show_gaia_panel = not self.show_gaia_panel
                elif button_key == 'player':
                    print("Abriendo perfil de jugador...")
                    self.current_screen = "profiles"
                    return None
                elif button_key == 'settings':
                    print("Abriendo configuración...")
                    return 'settings'
                elif button_key == 'help':
                    print("Abriendo ayuda...")
                    self.help_popup = create_help_popup(self.screen, "home")
                    self.help_popup.show()
                    return 'help'
                elif button_key == 'background':
                    # Cambiar la opacidad del fondo al hacer clic
                    self.background_opacity = (self.background_opacity + 50) % 256
                    print(f"Cambiando opacidad del fondo a: {self.background_opacity}")
                    return None
        return None

    def draw_profiles_screen(self):
        """Dibujar pantalla de perfiles con botones más pequeños"""
        # Dibujar fondo
        self.draw_animated_background()
        
        # Panel principal
        panel_width = 600 if not self.is_mobile else self.screen_width - 40
        panel_height = 500 if not self.is_mobile else self.screen_height - 100
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2
        
        # Fondo del panel con transparencia
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill(self.colors['panel_dark'])
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Borde del panel
        pygame.draw.rect(self.screen, self.colors['ice_blue'], 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Título
        title = self.font_title.render("PERFILES DE JUGADOR", True, self.colors['ice_blue'])
        title_rect = title.get_rect(center=(self.screen_width // 2, panel_y + 40))
        self.screen.blit(title, title_rect)
        
        # Lista de perfiles
        profiles_y_start = panel_y + 100
        profile_height = 60
        
        ui_elements = {
            'profiles': [],
            'create_button': None,
            'back_button': None,
            'skin_button': None,
            'load_button': None,
            'delete_button': None
        }
        
        # Mostrar perfiles existentes
        for i, profile in enumerate(self.profiles):
            y_pos = profiles_y_start + i * (profile_height + 10)
            
            # Verificar si es el perfil seleccionado
            is_selected = i == self.selected_profile_index
            
            # Fondo del perfil
            profile_rect = pygame.Rect(panel_x + 20, y_pos, panel_width - 40, profile_height)
            color = self.colors['button_hover'] if is_selected else (50, 50, 50, 180)
            pygame.draw.rect(self.screen, color, profile_rect, border_radius=5)
            
            # Borde del perfil
            border_color = self.colors['text_gold'] if is_selected else self.colors['ice_blue']
            pygame.draw.rect(self.screen, border_color, profile_rect, 2, border_radius=5)
            
            # Nombre del jugador
            name_text = self.font_subtitle.render(profile['player_name'], True, self.colors['text_white'])
            self.screen.blit(name_text, (profile_rect.x + 20, profile_rect.y + 10))
            
            # Puntos y fecha
            points_text = f"Puntos: {profile['total_points']}"
            date_text = f"Último acceso: {profile['last_played'][:10]}"
            
            points_surface = self.font_text.render(points_text, True, self.colors['text_gold'])
            date_surface = self.font_small.render(date_text, True, self.colors['text_white'])
            
            self.screen.blit(points_surface, (profile_rect.x + 20, profile_rect.y + 35))
            self.screen.blit(date_surface, (profile_rect.x + panel_width // 2, profile_rect.y + 38))
            
            ui_elements['profiles'].append({
                'rect': profile_rect,
                'index': i
            })
        
        # Botones de acción - Tamaños reducidos
        button_width = 100  # Ancho reducido
        button_height = 30  # Alto reducido
        button_y = panel_y + panel_height - 60  # Posición ajustada
        button_spacing = 10  # Espaciado entre botones
        
        # Botón Nuevo Perfil (más pequeño)
        create_button = pygame.Rect(panel_x + 20, button_y, button_width, button_height)
        pygame.draw.rect(self.screen, self.colors['hope_green'], create_button, border_radius=3)
        create_text = self.font_small.render("Nuevo", True, self.colors['text_white'])  # Texto más corto
        create_text_rect = create_text.get_rect(center=create_button.center)
        self.screen.blit(create_text, create_text_rect)
        ui_elements['create_button'] = create_button
        
        # Botón Cargar (solo si hay perfil seleccionado)
        if self.selected_profile_index >= 0:
            load_button = pygame.Rect(create_button.right + button_spacing, button_y, button_width, button_height)
            pygame.draw.rect(self.screen, self.colors['button_active'], load_button, border_radius=3)
            load_text = self.font_small.render("Cargar", True, self.colors['text_white'])
            load_text_rect = load_text.get_rect(center=load_button.center)
            self.screen.blit(load_text, load_text_rect)
            ui_elements['load_button'] = load_button
            
            # Botón Eliminar (texto abreviado a "Elim")
            delete_button = pygame.Rect(load_button.right + button_spacing, button_y, button_width, button_height)
            pygame.draw.rect(self.screen, self.colors['critical_red'], delete_button, border_radius=3)
            delete_text = self.font_small.render("Elim", True, self.colors['text_white'])
            delete_text_rect = delete_text.get_rect(center=delete_button.center)
            self.screen.blit(delete_text, delete_text_rect)
            ui_elements['delete_button'] = delete_button
            
            # Botón Skin (nuevo botón)
            skin_button = pygame.Rect(delete_button.right + button_spacing, button_y, button_width, button_height)
            pygame.draw.rect(self.screen, self.colors['purple'], skin_button, border_radius=3)
            skin_text = self.font_small.render("Skin", True, self.colors['text_white'])
            skin_text_rect = skin_text.get_rect(center=skin_button.center)
            self.screen.blit(skin_text, skin_text_rect)
            ui_elements['skin_button'] = skin_button
        
        # Botón Volver (ajustado a nuevo tamaño)
        back_button = pygame.Rect(panel_x + panel_width - button_width - 20, button_y, button_width, button_height)
        pygame.draw.rect(self.screen, self.colors['warning_orange'], back_button, border_radius=3)
        back_text = self.font_small.render("Volver", True, self.colors['text_white'])
        back_text_rect = back_text.get_rect(center=back_button.center)
        self.screen.blit(back_text, back_text_rect)
        ui_elements['back_button'] = back_button
        
        # Mostrar mensajes
        self.draw_messages()
        
        return ui_elements

    def draw_create_profile_screen(self):
        """Dibujar pantalla de creación de perfil"""
        # Dibujar fondo
        self.draw_animated_background()
        
        # Panel de creación
        panel_width = 400 if not self.is_mobile else self.screen_width - 40
        panel_height = 300
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2
        
        # Fondo del panel
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill(self.colors['panel_dark'])
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Borde del panel
        pygame.draw.rect(self.screen, self.colors['hope_green'], 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Título
        title = self.font_subtitle.render("CREAR NUEVO PERFIL", True, self.colors['hope_green'])
        title_rect = title.get_rect(center=(self.screen_width // 2, panel_y + 40))
        self.screen.blit(title, title_rect)
        
        # Campo de entrada
        input_rect = pygame.Rect(panel_x + 50, panel_y + 100, panel_width - 100, 40)
        color = self.colors['text_white'] if self.input_active else (100, 100, 100)
        pygame.draw.rect(self.screen, color, input_rect, 2)
        
        # Texto del input
        text_surface = self.font_text.render(self.input_text, True, self.colors['text_white'])
        self.screen.blit(text_surface, (input_rect.x + 10, input_rect.y + 10))
        
        # Cursor parpadeante
        if self.input_active and int(self.animation_time * 2) % 2:
            cursor_x = input_rect.x + 10 + text_surface.get_width()
            pygame.draw.line(self.screen, self.colors['text_white'], 
                           (cursor_x, input_rect.y + 5), 
                           (cursor_x, input_rect.y + 35), 2)
        
        # Instrucciones
        instruction = self.font_small.render("Ingresa tu nombre de jugador", True, self.colors['text_white'])
        instruction_rect = instruction.get_rect(center=(self.screen_width // 2, panel_y + 80))
        self.screen.blit(instruction, instruction_rect)
        
        # Botones
        button_y = panel_y + panel_height - 60
        
        # Botón Crear
        create_button = pygame.Rect(panel_x + 50, button_y, 100, 40)
        pygame.draw.rect(self.screen, self.colors['hope_green'], create_button, border_radius=5)
        create_text = self.font_text.render("Crear", True, self.colors['text_white'])
        create_text_rect = create_text.get_rect(center=create_button.center)
        self.screen.blit(create_text, create_text_rect)
        
        # Botón Cancelar
        cancel_button = pygame.Rect(panel_x + panel_width - 150, button_y, 100, 40)
        pygame.draw.rect(self.screen, self.colors['critical_red'], cancel_button, border_radius=5)
        cancel_text = self.font_text.render("Cancelar", True, self.colors['text_white'])
        cancel_text_rect = cancel_text.get_rect(center=cancel_button.center)
        self.screen.blit(cancel_text, cancel_text_rect)
        
        # Mostrar mensajes
        self.draw_messages()
        
        return {
            'input_rect': input_rect,
            'create_button': create_button,
            'cancel_button': cancel_button
        }

    def handle_profiles_events(self, event, ui_elements):
        """Manejar eventos en la pantalla de perfiles"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Clic izquierdo
                # Verificar clic en perfiles
                for profile_data in ui_elements['profiles']:
                    if profile_data['rect'].collidepoint(event.pos):
                        self.selected_profile_index = profile_data['index']
                        break
                
                # Verificar botones
                if ui_elements['create_button'].collidepoint(event.pos):
                    self.current_screen = "create_profile"
                    self.input_text = ""
                    self.input_active = True
                
                elif ui_elements['back_button'].collidepoint(event.pos):
                    self.current_screen = "main"
                
                elif 'load_button' in ui_elements and ui_elements['load_button'].collidepoint(event.pos):
                    if self.load_selected_profile():
                        self.current_screen = "main"
                
                elif 'delete_button' in ui_elements and ui_elements['delete_button'].collidepoint(event.pos):
                    self.delete_selected_profile()
                
                elif 'skin_button' in ui_elements and ui_elements['skin_button'].collidepoint(event.pos):
                    self.current_screen = "skin_selection"  # Nueva pantalla para selección de skins
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.current_screen = "main"

    def draw_skin_selection_screen(self):
        """Dibujar pantalla de selección de skins con vista previa interactiva"""
        # Dibujar fondo
        self.draw_animated_background()
        
        # Panel principal
        panel_width = 700 if not self.is_mobile else self.screen_width - 40
        panel_height = 550 if not self.is_mobile else self.screen_height - 100
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2
        
        # Fondo del panel con transparencia
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((20, 20, 40, 220))  # Fondo más oscuro para mejor contraste
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Borde del panel con efecto neón
        pygame.draw.rect(self.screen, self.colors['purple'], 
                        (panel_x, panel_y, panel_width, panel_height), 3, border_radius=10)
        
        # Título con efecto especial
        title = self.font_title.render("SELECCIONAR SKIN", True, self.colors['ice_blue'])
        title_shadow = self.font_title.render("SELECCIONAR SKIN", True, (100, 100, 255, 150))
        title_rect = title.get_rect(center=(self.screen_width // 2, panel_y + 50))
        self.screen.blit(title_shadow, (title_rect.x + 3, title_rect.y + 3))
        self.screen.blit(title, title_rect)
        
        # Mostrar nombre del perfil actual
        if 0 <= self.selected_profile_index < len(self.profiles):
            profile_name = self.profiles[self.selected_profile_index]['player_name']
            subtitle = self.font_subtitle.render(f"Perfil: {profile_name}", True, self.colors['text_white'])
            subtitle_rect = subtitle.get_rect(center=(self.screen_width // 2, panel_y + 90))
            self.screen.blit(subtitle, subtitle_rect)
        
        # Definir las skins disponibles (mejor mover esto a __init__)
        available_skins = [
            {'id': 'default', 'name': 'Predeterminado', 'color': (200, 200, 200)},
            {'id': 'eco_warrior', 'name': 'Guerrero Eco', 'color': (100, 200, 100)},
            {'id': 'arctic', 'name': 'Explorador Ártico', 'color': (150, 220, 255)},
            {'id': 'volcano', 'name': 'Resistente Volcánico', 'color': (255, 100, 50)},
            {'id': 'cyber', 'name': 'Hacker Climático', 'color': (100, 255, 200)},
            {'id': 'retro', 'name': 'Retro Salvador', 'color': (255, 200, 100)},
            {'id': 'scientist', 'name': 'Científico', 'color': (200, 200, 255)},
            {'id': 'agent', 'name': 'Agente Especial', 'color': (50, 50, 100)},
        ]
        
        # Obtener skin actual
        current_skin_id = getattr(self, 'selected_skin_id', 'default')
        if 0 <= self.selected_profile_index < len(self.profiles):
            current_skin_id = self.profiles[self.selected_profile_index].get('skin', current_skin_id)
        
        # Área de vista previa (derecha)
        preview_width = 250
        preview_x = panel_x + panel_width - preview_width - 30
        preview_y = panel_y + 140
        preview_height = panel_height - 200
        
        # Fondo de vista previa
        pygame.draw.rect(self.screen, (30, 30, 60), (preview_x, preview_y, preview_width, preview_height), border_radius=15)
        pygame.draw.rect(self.screen, self.colors['ice_blue'], (preview_x, preview_y, preview_width, preview_height), 2, border_radius=15)
        
        # Título "Vista Previa"
        preview_title = self.font_subtitle.render("VISTA PREVIA", True, self.colors['text_white'])
        self.screen.blit(preview_title, (preview_x + preview_width//2 - preview_title.get_width()//2, preview_y + 20))
        
        # Dibujar skin seleccionada en vista previa (con la forma específica)
        preview_radius = 80
        preview_center_x = preview_x + preview_width // 2
        preview_center_y = preview_y + preview_height // 2
        
        # Crear superficie para la skin
        skin_surface = pygame.Surface((preview_radius * 2, preview_radius * 2), pygame.SRCALPHA)
        skin_color = next((sk['color'] for sk in available_skins if sk['id'] == current_skin_id), (200, 200, 200))
        
        # Dibujar círculo principal
        pygame.draw.circle(skin_surface, skin_color, (preview_radius, preview_radius), preview_radius)
        
        # Añadir brillo al centro exactamente como lo especificaste
        pygame.draw.circle(skin_surface, (255, 255, 255, 150), (preview_radius, preview_radius), preview_radius // 2)
        
        # Dibujar en pantalla
        self.screen.blit(skin_surface, (preview_center_x - preview_radius, preview_center_y - preview_radius))
        
        # Nombre de la skin actual
        skin_name = next((sk['name'] for sk in available_skins if sk['id'] == current_skin_id), "Predeterminado")
        selected_text = self.font_text.render(f"Skin actual: {skin_name}", True, self.colors['text_gold'])
        self.screen.blit(selected_text, (preview_center_x - selected_text.get_width()//2, preview_center_y + preview_radius + 20))

        # Lista de skins (izquierda)
        skins_y_start = panel_y + 140
        skin_size = 70
        skins_per_row = 3
        padding = 25
        max_skins = 8
        
        ui_elements = {
            'skins': [],
            'back_button': None
        }
        
        # Organizar skins en 2 columnas
        for i, skin in enumerate(available_skins[:max_skins]):
            row = i // skins_per_row
            col = i % skins_per_row
            
            x_pos = panel_x + padding + col * (skin_size + padding + 40)
            y_pos = skins_y_start + row * (skin_size + padding + 10)
            
            skin_rect = pygame.Rect(x_pos, y_pos, skin_size, skin_size)
            
            # Fondo del ícono de skin
            pygame.draw.circle(self.screen, (40, 40, 70), (x_pos + skin_size//2, y_pos + skin_size//2), skin_size//2 + 5)
            
            # Ícono de skin
            pygame.draw.circle(self.screen, skin['color'], (x_pos + skin_size//2, y_pos + skin_size//2), skin_size//2 - 3)
            
            # Resaltar selección actual
            if skin['id'] == current_skin_id:
                pygame.draw.circle(self.screen, self.colors['text_gold'], 
                                (x_pos + skin_size//2, y_pos + skin_size//2), 
                                skin_size//2 + 2, 3)
                # Efecto de selección
                glow_surface = pygame.Surface((skin_size+10, skin_size+10), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*self.colors['text_gold'], 80), 
                                (skin_size//2 + 5, skin_size//2 + 5), skin_size//2 + 5)
                self.screen.blit(glow_surface, (x_pos - 5, y_pos - 5))
            
            # Nombre de la skin
            name_text = self.font_small.render(skin['name'], True, self.colors['text_white'])
            name_rect = name_text.get_rect(center=(x_pos + skin_size//2, y_pos + skin_size + 15))
            self.screen.blit(name_text, name_rect)
            
            ui_elements['skins'].append({
                'rect': skin_rect,
                'skin_id': skin['id']
            })
        
        # Botón Volver con mejor diseño
        button_width = 100
        button_height = 30
        back_button = pygame.Rect(panel_x + panel_width - button_width - 30, 
                                panel_y + panel_height - button_height - 20, 
                                button_width, button_height)
        
        # Efecto de botón
        pygame.draw.rect(self.screen, (80, 40, 0), back_button, border_radius=5)
        pygame.draw.rect(self.screen, self.colors['warning_orange'], back_button, border_radius=5)
        pygame.draw.rect(self.screen, (255, 180, 50), back_button, 2, border_radius=5)
        
        back_text = self.font_text.render("VOLVER", True, self.colors['text_white'])
        back_text_rect = back_text.get_rect(center=back_button.center)
        self.screen.blit(back_text, back_text_rect)
        ui_elements['back_button'] = back_button
        
        # Mostrar mensajes
        self.draw_messages()
        
        return ui_elements

    def handle_skin_selection_events(self, event, ui_elements):
        """Manejar eventos en la pantalla de selección de skins"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Verificar clic en skins
            for skin_data in ui_elements['skins']:
                if skin_data['rect'].collidepoint(event.pos):
                    self.selected_skin_id = skin_data['skin_id']
                    # Actualizar vista previa inmediatamente
                    return self.draw_skin_selection_screen()
            
            # Verificar botón volver
            if ui_elements['back_button'].collidepoint(event.pos):           
                self.current_screen = "profiles"
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.current_screen = "profiles"

    def handle_create_profile_events(self, event, ui_elements):
        """Manejar eventos en la pantalla de creación de perfil"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Activar/desactivar campo de texto
                if ui_elements['input_rect'].collidepoint(event.pos):
                    self.input_active = True
                else:
                    self.input_active = False
                
                # Verificar botones
                if ui_elements['create_button'].collidepoint(event.pos):
                    if self.create_new_profile():
                        self.current_screen = "main"
                
                elif ui_elements['cancel_button'].collidepoint(event.pos):
                    self.current_screen = "profiles"
                    self.input_text = ""
        
        elif event.type == pygame.KEYDOWN:
            if self.input_active:
                if event.key == pygame.K_RETURN:
                    if self.create_new_profile():
                        self.current_screen = "main"
                elif event.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    self.current_screen = "profiles"
                    self.input_text = ""
                else:
                    # Limitar longitud del nombre
                    if len(self.input_text) < 20:
                        self.input_text += event.unicode

    def draw_messages(self):
        """Dibujar mensajes de error o éxito"""
        current_time = time.time()
        
        # Mostrar mensaje de error
        if self.error_message and (current_time - self.message_time) < 3:
            error_surface = self.font_text.render(self.error_message, True, self.colors['critical_red'])
            error_rect = error_surface.get_rect(center=(self.screen_width // 2, self.screen_height - 100))
            
            # Fondo para el mensaje
            bg_rect = pygame.Rect(error_rect.x - 10, error_rect.y - 5, 
                                error_rect.width + 20, error_rect.height + 10)
            pygame.draw.rect(self.screen, (40, 0, 0), bg_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.colors['critical_red'], bg_rect, 2, border_radius=5)
            
            self.screen.blit(error_surface, error_rect)
        
        # Mostrar mensaje de éxito
        if self.success_message and (current_time - self.message_time) < 3:
            success_surface = self.font_text.render(self.success_message, True, self.colors['hope_green'])
            success_rect = success_surface.get_rect(center=(self.screen_width // 2, self.screen_height - 100))
            
            # Fondo para el mensaje
            bg_rect = pygame.Rect(success_rect.x - 10, success_rect.y - 5, 
                                success_rect.width + 20, success_rect.height + 10)
            pygame.draw.rect(self.screen, (0, 40, 0), bg_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.colors['hope_green'], bg_rect, 2, border_radius=5)
            
            self.screen.blit(success_surface, success_rect)

    def draw_active_profile(self):
        """Dibujar el perfil activo en la esquina inferior izquierda"""
        if self.save_system.current_profile:
            profile_name = self.save_system.current_profile["player_name"]
            profile_text = f"Agente: {profile_name}"
            
            # Crear superficie para el fondo del mensaje
            profile_surface = self.font_text.render(profile_text, True, self.colors['text_gold'])
            profile_rect = profile_surface.get_rect()
            profile_rect.bottomleft = (20, self.screen_height - 20)
            
            # Dibujar fondo semi-transparente
            bg_rect = profile_rect.inflate(20, 10)
            bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surface.fill((20, 20, 40, 200))
            self.screen.blit(bg_surface, bg_rect)
            
            # Dibujar borde
            pygame.draw.rect(self.screen, self.colors['ice_blue'], bg_rect, 1)
            
            # Dibujar texto
            self.screen.blit(profile_surface, profile_rect)

    def run(self):
        """Bucle principal del menú"""
        running = True
        result = None

        # Crear botones
        play_button = Button('play', self.buttons['play']['scale'], self.buttons['play']['pos'],  self.buttons['play']['tex_hover'])
        history_button = Button('history', self.buttons['history']['scale'], self.buttons['history']['pos'],  self.buttons['history']['tex_hover'])
        player_button = Button('player', self.buttons['player']['scale'], self.buttons['player']['pos'],  self.buttons['player']['tex_hover'])
        settings_button = Button('settings', self.buttons['settings']['scale'], self.buttons['settings']['pos'],  self.buttons['settings']['tex_hover'])
        help_button = Button('help', self.buttons['help']['scale'], self.buttons['help']['pos'],  self.buttons['help']['tex_hover'])
        
        while running:
            dt = self.clock.tick(60) / 1000.0
            self.animation_time += dt
            
            # Actualizar efectos ambientales
            self.update_environmental_effects(dt)
            
            # Actualizar popup si existe
            if self.help_popup:
                popup_result = self.help_popup.update(dt)
                if popup_result == "closed":
                    self.help_popup = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    result = "exit"  # Añadido para señalar salida del juego
                
                # Manejar eventos del popup si está visible
                if self.help_popup and self.help_popup.is_visible():
                    popup_action = self.help_popup.handle_event(event)
                    if popup_action == "more_info":
                        # Aquí podrías implementar alguna acción adicional
                        pass
                    continue  # Si el popup está visible, no procesar otros eventos

                # Manejo de eventos según la pantalla actual
                if self.current_screen == "main":
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Clic izquierdo
                            action = self.handle_click(event.pos)
                            if action == 'level_select':  # Cambiado para manejar transición a selección de niveles
                                # Guardar último acceso antes de cambiar de pantalla
                                if self.save_system.current_profile:
                                    self.save_system.save_current_profile()
                                result = action
                                running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            action = self.handle_click(self.buttons['play']['pos'])
                            if action == 'level_select':  # Cambiado para manejar transición a selección de niveles
                                result = action
                                running = False
                
                elif self.current_screen == "profiles":
                    ui_elements = self.draw_profiles_screen()
                    self.handle_profiles_events(event, ui_elements)
                    
                elif self.current_screen == "create_profile":
                    ui_elements = self.draw_create_profile_screen()
                    self.handle_create_profile_events(event, ui_elements)
            
            # Dibujar según la pantalla current_screen
            if self.current_screen == "main":
                # Dibujar pantalla principal
                self.draw_animated_background()
                self.draw_title()
                
                # Dibujar botones with efectos hover
                self.draw_circular_button(play_button)
                self.draw_circular_button(history_button)
                self.draw_circular_button(player_button)
                self.draw_circular_button(settings_button)
                self.draw_circular_button(help_button)
                
                # Dibujar paneles informativos
                self.draw_gaia_panel()
                self.draw_progress_panel()
                self.draw_climate_warning()
                self.draw_active_profile()

                # Dibujar popup de ayuda si está visible
                if self.help_popup:
                    self.help_popup.draw()
                
            elif self.current_screen == "profiles":
                self.draw_profiles_screen()
                
            elif self.current_screen == "create_profile":
                self.draw_create_profile_screen()

            elif self.current_screen == "skin_selection":
                ui_elements = self.draw_skin_selection_screen()
                self.handle_skin_selection_events(event, ui_elements)
            
            # Mostrar mensajes generales   
            self.draw_messages()            
            
            pygame.display.flip()
        
        return result if result else "exit"  # Aseguramos que siempre devuelva un valor

