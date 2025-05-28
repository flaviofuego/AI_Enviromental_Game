import pygame
import sys
import math
import json
import os, random
from datetime import datetime
import time

from ..config.save_system import GameSaveSystem
from ..components.Button import Button

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
        pygame.display.set_caption("Hockey Is Melting Down - Salva la Tierra")
        
        # Cargar imagen de fondo
        self.background_image = pygame.image.load('game/assets/background.png')
        self.background_opacity = 200  # 0-255, donde 255 es completamente opaco
        
        # Inicializar sistema de guardado
        self.save_system = GameSaveSystem()
        
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
                'play': {'pos': (center_x, center_y - 80), 'radius': 100, 'color': self.colors['hope_green']},
                'history': {'pos': (center_x - 80, center_y + 40), 'radius': 30, 'color': self.colors['ice_blue']},
                'player': {'pos': (center_x + 80, center_y + 40), 'radius': 30, 'color': self.colors['warning_orange']},
                'settings': {'pos': (30, 30), 'radius': 30, 'color': self.colors['ice_blue']},
                'help': {'pos': (self.screen_width - 30, 30), 'radius': 30, 'color': self.colors['ice_blue']}
            }
        else:
            # Disposición para PC
            self.buttons = {
                'play': {'pos': (center_x, center_y), 'radius': 300, 'color': self.colors['hope_green']},
                'history': {'pos': (center_x - 420, center_y), 'radius': 120, 'color': self.colors['ice_blue']},
                'player': {'pos': (center_x + 420, center_y), 'radius': 120, 'color': self.colors['warning_orange']},
                'background': {'pos': (center_x, center_y + 120), 'radius': 30, 'color': self.colors['button_active']},
                'settings': {'pos': (40, 40), 'radius': 80, 'color': self.colors['ice_blue']},
                'help': {'pos': (self.screen_width - 40, 40), 'radius': 80, 'color': self.colors['ice_blue']}
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
        
        # Cargar perfiles al iniciar
        self.load_profiles()
        
        self.clock = pygame.time.Clock()
    
    def load_profiles(self):
        """Carga la lista de perfiles disponibles"""
        self.profiles = self.save_system.get_all_profiles()
        
        # Si hay perfiles disponibles, pre-seleccionar el primero (el más reciente)
        if self.profiles:
            self.selected_profile_index = 0
    
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
        self.draw_gradient_background()
        
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
    
    def draw_circular_button(self, button_key, icon_text, hover_text=""):
        """Dibujar botón circular con efectos"""
        button = self.buttons[button_key]
        pos = button['pos']
        radius = button['radius']
        color = button['color']

        draw_button = Button(button_key, (radius, radius), pos)
        draw_button.draw(self.screen)
        is_hovered = draw_button.is_hovered()
        
        # Mostrar texto de hover
        if is_hovered and hover_text:
            hover_surface = self.font_small.render(hover_text, True, self.colors['text_white'])
            hover_rect = hover_surface.get_rect(center=(pos[0], pos[1] + radius + 10))
            self.screen.blit(hover_surface, hover_rect)

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
            distance = math.sqrt((pos[0] - button['pos'][0])**2 + (pos[1] - button['pos'][1])**2)
            if distance <= button['radius']:
                if button_key == 'play':
                    print("Iniciando juego...")
                    
                    # Verificar si hay un perfil activo
                    if not self.save_system.current_profile:
                        self.current_screen = "profiles"
                        self.error_message = "Selecciona o crea un perfil para jugar"
                        self.message_time = time.time()
                        return None
                    
                    return 'start_game'
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
                    return 'help'
                elif button_key == 'background':
                    # Cambiar la opacidad del fondo al hacer clic
                    self.background_opacity = (self.background_opacity + 50) % 256
                    print(f"Cambiando opacidad del fondo a: {self.background_opacity}")
                    return None
        return None
    
    def run(self):
        """Bucle principal del menú"""
        running = True
        
        while running:
            dt = self.clock.tick(60) / 1000.0
            self.animation_time += dt
            
            # Actualizar efectos ambientales
            self.update_environmental_effects(dt)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Manejo de eventos según la pantalla actual
                if self.current_screen == "main":
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Clic izquierdo
                            action = self.handle_click(event.pos)
                            if action == 'start_game':
                                # Guardar último acceso antes de iniciar el juego
                                if self.save_system.current_profile:
                                    self.save_system.save_current_profile()
                                running = False  # Salir para iniciar el juego
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            action = self.handle_click(self.buttons['play']['pos'])
                            if action == 'start_game':
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
                self.draw_circular_button('play', '▶', "Comenzar Misión")
                self.draw_circular_button('history', '📜', "Historial de Partidas")
                self.draw_circular_button('player', '👤', "Personalizar Jugador")
                self.draw_circular_button('settings', '⚙', "Configuración")
                self.draw_circular_button('help', '?', "Ayuda")
                
                if not self.is_mobile:
                    #self.draw_circular_button('background', '🎨', "Fondo Temático")
                    pass
                
                # Dibujar paneles informativos
                self.draw_gaia_panel()
                self.draw_progress_panel()
                self.draw_climate_warning()
                
                # Mostrar mensaje del perfil activo
                if self.save_system.current_profile:
                    profile_name = self.save_system.current_profile["player_name"]
                    profile_text = f"Perfil: {profile_name}"
                    profile_surface = self.font_text.render(profile_text, True, self.colors['text_gold'])
                    self.screen.blit(profile_surface, (10, 10))
                
            """  elif self.current_screen == "profiles":
                self.draw_profiles_screen()
                
            elif self.current_screen == "create_profile":
                self.draw_create_profile_screen() """
            
            # Mostrar mensajes generales
            
            
            pygame.display.flip()
        
        pygame.quit()
        return "start_game"

