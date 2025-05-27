import pygame
import sys
import math
import random

# Inicializar Pygame
pygame.init()
pygame.mixer.init()

# Configuración de pantalla
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
FPS = 60

# Colores temáticos ambientales
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
OCEAN_BLUE = (30, 130, 200)
ICE_BLUE = (173, 216, 230)
TOXIC_GREEN = (100, 200, 50)
POLLUTION_GRAY = (80, 80, 80)
FIRE_RED = (220, 50, 50)
EARTH_BROWN = (139, 69, 19)
SKY_BLUE = (135, 206, 235)
GAIA_GREEN = (34, 139, 34)
WARNING_ORANGE = (255, 140, 0)

class EnvironmentalParticle:
    """Partículas temáticas ambientales"""
    def __init__(self, x, y, particle_type="ice"):
        self.x = x
        self.y = y
        self.type = particle_type
        self.speed = random.uniform(0.5, 2)
        self.size = random.randint(2, 6)
        self.angle = random.uniform(0, 2 * math.pi)
        self.life = 255
        self.decay = random.randint(1, 3)
        
        # Colores según el tipo de partícula
        if particle_type == "ice":
            self.color = random.choice([ICE_BLUE, WHITE, (200, 230, 255)])
        elif particle_type == "pollution":
            self.color = random.choice([POLLUTION_GRAY, (60, 60, 60), (100, 100, 100)])
        elif particle_type == "fire":
            self.color = random.choice([FIRE_RED, WARNING_ORANGE, (255, 100, 0)])
    
    def update(self):
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed * 0.5
        self.life -= self.decay
        if self.life < 0:
            self.life = 0
    
    def draw(self, screen):
        if self.life > 0:
            alpha = max(0, self.life)
            color_with_alpha = (*self.color[:3], alpha)
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)

class CircularButton:
    """Botón circular temático"""
    def __init__(self, x, y, radius, icon, color, hover_color, tooltip=""):
        self.x = x
        self.y = y
        self.radius = radius
        self.icon = icon
        self.color = color
        self.hover_color = hover_color
        self.tooltip = tooltip
        self.hovered = False
        self.font = pygame.font.Font(None, 24)
        self.icon_font = pygame.font.Font(None, 36)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            distance = math.sqrt((event.pos[0] - self.x)**2 + (event.pos[1] - self.y)**2)
            self.hovered = distance <= self.radius
        elif event.type == pygame.MOUSEBUTTONDOWN:
            distance = math.sqrt((event.pos[0] - self.x)**2 + (event.pos[1] - self.y)**2)
            if distance <= self.radius:
                return True
        return False
    
    def draw(self, screen):
        color = self.hover_color if self.hovered else self.color
        pygame.draw.circle(screen, color, (self.x, self.y), self.radius)
        pygame.draw.circle(screen, WHITE, (self.x, self.y), self.radius, 3)
        
        # Dibujar icono
        icon_surface = self.icon_font.render(self.icon, True, WHITE)
        icon_rect = icon_surface.get_rect(center=(self.x, self.y))
        screen.blit(icon_surface, icon_rect)
        
        # Tooltip
        if self.hovered and self.tooltip:
            tooltip_surface = self.font.render(self.tooltip, True, WHITE)
            tooltip_rect = tooltip_surface.get_rect(center=(self.x, self.y + self.radius + 20))
            pygame.draw.rect(screen, BLACK, tooltip_rect.inflate(10, 5))
            screen.blit(tooltip_surface, tooltip_rect)

class PlayButton:
    """Botón de jugar central con diseño especial"""
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.hovered = False
        self.pulse = 0
        self.font = pygame.font.Font(None, 48)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            distance = math.sqrt((event.pos[0] - self.x)**2 + (event.pos[1] - self.y)**2)
            self.hovered = distance <= self.radius
        elif event.type == pygame.MOUSEBUTTONDOWN:
            distance = math.sqrt((event.pos[0] - self.x)**2 + (event.pos[1] - self.y)**2)
            if distance <= self.radius:
                return True
        return False
    
    def update(self, time):
        self.pulse = math.sin(time * 0.1) * 10
    
    def draw(self, screen):
        # Efecto de pulso
        current_radius = self.radius + (self.pulse if self.hovered else 0)
        
        # Círculo exterior (GaiaCore)
        pygame.draw.circle(screen, GAIA_GREEN, (self.x, self.y), int(current_radius))
        pygame.draw.circle(screen, WHITE, (self.x, self.y), int(current_radius), 4)
        
        # Triángulo de play
        triangle_size = current_radius // 3
        points = [
            (self.x + triangle_size//2, self.y),
            (self.x - triangle_size//2, self.y - triangle_size//2),
            (self.x - triangle_size//2, self.y + triangle_size//2)
        ]
        pygame.draw.polygon(screen, WHITE, points)

class LorePanel:
    """Panel con información del lore"""
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 32)
        self.scroll_offset = 0
        
        self.lore_text = [
            "GAIA CORE - SISTEMA DE RESTAURACIÓN",
            "",
            "La Tierra agoniza. Los polos se derriten,",
            "los bosques desaparecen y las ciudades",
            "arden bajo olas de calor implacables.",
            "",
            "Cinco agentes corruptos de EcoNull",
            "han tomado control del clima:",
            "",
            "🌊 SLICKWAVE - Emperador del plástico",
            "   Inunda los océanos con desechos",
            "",
            "☀️ UVBLADE - Destructor del ozono", 
            "   Ha perforado el escudo celestial",
            "",
            "💨 SMOGATRON - Señor de la niebla tóxica",
            "   Asfixia las ciudades con veneno",
            "",
            "🌳 DEFORESTIX - Talador de raíces",
            "   Devora los pulmones del planeta",
            "",
            "🔥 HEATCORE - Maestro del calor urbano",
            "   Convierte las urbes en hornos",
            "",
            "Tu misión como último guardián:",
            "Derrotar a cada agente en duelos",
            "de hockey sobre hielo sagrado.",
            "",
            "Cada victoria restaura el equilibrio",
            "y devuelve la esperanza a la Tierra."
        ]
    
    def draw(self, screen):
        # Fondo semi-transparente
        pygame.draw.rect(screen, (0, 0, 0, 180), self.rect)
        pygame.draw.rect(screen, GAIA_GREEN, self.rect, 2)
        
        # Título
        title_surface = self.title_font.render("ARCHIVO GAIA", True, GAIA_GREEN)
        title_rect = title_surface.get_rect(centerx=self.rect.centerx, y=self.rect.y + 10)
        screen.blit(title_surface, title_rect)
        
        # Texto del lore
        y_offset = self.rect.y + 45
        for line in self.lore_text:
            if y_offset < self.rect.bottom - 20:
                if line.startswith(("🌊", "☀️", "💨", "🌳", "🔥")):
                    color = WARNING_ORANGE
                elif line == "GAIA CORE - SISTEMA DE RESTAURACIÓN":
                    color = GAIA_GREEN
                else:
                    color = WHITE
                
                text_surface = self.font.render(line, True, color)
                screen.blit(text_surface, (self.rect.x + 10, y_offset))
                y_offset += 25

class ScorePanel:
    """Panel de puntuación y estadísticas ambientales"""
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        
        # Estadísticas ambientales simuladas
        self.stats = {
            "Océanos limpiados": "12%",
            "Ozono restaurado": "5%", 
            "Aire purificado": "8%",
            "Bosques replantados": "15%",
            "Ciudades enfriadas": "3%",
            "Puntos Gaia": "1,250"
        }
    
    def draw(self, screen):
        # Fondo
        pygame.draw.rect(screen, (0, 50, 100, 200), self.rect)
        pygame.draw.rect(screen, ICE_BLUE, self.rect, 2)
        
        # Título
        title_surface = self.font.render("PROGRESO PLANETARIO", True, ICE_BLUE)
        title_rect = title_surface.get_rect(centerx=self.rect.centerx, y=self.rect.y + 10)
        screen.blit(title_surface, title_rect)
        
        # Estadísticas
        y_offset = self.rect.y + 45
        for stat, value in self.stats.items():
            stat_surface = self.small_font.render(f"{stat}:", True, WHITE)
            screen.blit(stat_surface, (self.rect.x + 10, y_offset))
            
            value_surface = self.small_font.render(value, True, GAIA_GREEN)
            value_rect = value_surface.get_rect(right=self.rect.right - 10, y=y_offset)
            screen.blit(value_surface, value_rect)
            
            y_offset += 25

class MainMenu:
    """Clase principal del menú mejorado"""
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Hockey Ice Melting Down - Salva la Tierra")
        self.clock = pygame.time.Clock()
        
        self.state = "MAIN"
        self.running = True
        self.time = 0
        
        # Partículas ambientales por zonas
        self.particles = []
        self.create_environmental_particles()
        
        # Fuentes
        self.title_font = pygame.font.Font(None, 64)
        self.subtitle_font = pygame.font.Font(None, 32)
        self.text_font = pygame.font.Font(None, 28)
        
        # Botón de jugar central
        self.play_button = PlayButton(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50, 80)
        
        # Botones circulares temáticos
        center_x, center_y = SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50
        radius_distance = 150
        
        self.circular_buttons = [
            CircularButton(center_x - radius_distance, center_y, 40, "🎵", OCEAN_BLUE, SKY_BLUE, "Música"),
            CircularButton(center_x + radius_distance, center_y, 40, "❓", ICE_BLUE, WHITE, "Ayuda"),
            CircularButton(center_x, center_y - radius_distance, 40, "👤", GAIA_GREEN, TOXIC_GREEN, "Perfil"),
            CircularButton(center_x, center_y + radius_distance, 40, "⚙️", POLLUTION_GRAY, WHITE, "Config")
        ]
        
        # Paneles informativos
        self.lore_panel = LorePanel(50, 150, 300, 400)
        self.score_panel = ScorePanel(SCREEN_WIDTH - 250, 150, 200, 300)
        
        # Sistema de mensajes ambientales
        self.environmental_messages = [
            "Los glaciares pierden 280 mil millones de toneladas anuales",
            "Cada minuto desaparecen 40 campos de fútbol de bosque",
            "8 millones de toneladas de plástico llegan al océano cada año",
            "La temperatura global ha subido 1.1°C desde 1880",
            "Solo el 3% del agua del planeta es dulce"
        ]
        self.current_message = 0
        self.message_timer = 0
        
    def create_environmental_particles(self):
        # Zona izquierda: partículas de hielo
        for _ in range(20):
            x = random.randint(0, SCREEN_WIDTH//3)
            y = random.randint(0, SCREEN_HEIGHT)
            self.particles.append(EnvironmentalParticle(x, y, "ice"))
        
        # Zona derecha: partículas de contaminación
        for _ in range(15):
            x = random.randint(2*SCREEN_WIDTH//3, SCREEN_WIDTH)
            y = random.randint(0, SCREEN_HEIGHT)
            self.particles.append(EnvironmentalParticle(x, y, "pollution"))
        
        # Zona superior: partículas de fuego
        for _ in range(10):
            x = random.randint(SCREEN_WIDTH//3, 2*SCREEN_WIDTH//3)
            y = random.randint(0, SCREEN_HEIGHT//3)
            self.particles.append(EnvironmentalParticle(x, y, "fire"))
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # Botón de jugar
            if self.play_button.handle_event(event):
                print("🌍 Iniciando misión de salvamento planetario...")
                # Aquí iniciarías el primer nivel
            
            # Botones circulares
            for i, button in enumerate(self.circular_buttons):
                if button.handle_event(event):
                    if i == 0:  # Música
                        print("🎵 Configuración de audio")
                    elif i == 1:  # Ayuda
                        print("❓ Sistema de ayuda GaiaCore")
                    elif i == 2:  # Perfil
                        print("👤 Perfil del guardián")
                    elif i == 3:  # Configuración
                        print("⚙️ Configuración del sistema")
    
    def update(self):
        self.time += 1
        self.play_button.update(self.time)
        
        # Actualizar partículas ambientales
        for particle in self.particles:
            particle.update()
            if particle.life <= 0 or particle.x < -50 or particle.x > SCREEN_WIDTH + 50:
                # Regenerar partícula según zona
                if particle.type == "ice":
                    particle.x = random.randint(0, SCREEN_WIDTH//3)
                elif particle.type == "pollution":
                    particle.x = random.randint(2*SCREEN_WIDTH//3, SCREEN_WIDTH)
                else:
                    particle.x = random.randint(SCREEN_WIDTH//3, 2*SCREEN_WIDTH//3)
                particle.y = random.randint(0, SCREEN_HEIGHT)
                particle.life = 255
        
        # Cambiar mensaje ambiental
        self.message_timer += 1
        if self.message_timer > 300:  # 5 segundos a 60 FPS
            self.current_message = (self.current_message + 1) % len(self.environmental_messages)
            self.message_timer = 0
    
    def draw_background(self):
        # Gradiente ambiental (cielo contaminado a océano)
        for y in range(SCREEN_HEIGHT):
            ratio = y / SCREEN_HEIGHT
            if ratio < 0.3:  # Cielo contaminado
                r = int(100 + (150 - 100) * (ratio / 0.3))
                g = int(50 + (100 - 50) * (ratio / 0.3))
                b = int(50 + (120 - 50) * (ratio / 0.3))
            else:  # Océano
                ocean_ratio = (ratio - 0.3) / 0.7
                r = int(150 - 120 * ocean_ratio)
                g = int(100 + 30 * ocean_ratio)
                b = int(120 + 80 * ocean_ratio)
            
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))
        
        # Dibujar partículas ambientales
        for particle in self.particles:
            particle.draw(self.screen)
    
    def draw_main_interface(self):
        self.draw_background()
        
        # Título principal con efecto de crisis
        title_y = 80 + math.sin(self.time * 0.05) * 5
        
        # Sombra del título
        shadow_surface = self.title_font.render("HOCKEY ICE", True, BLACK)
        shadow_rect = shadow_surface.get_rect(center=(SCREEN_WIDTH//2 + 3, title_y + 3))
        self.screen.blit(shadow_surface, shadow_rect)
        
        # Título principal
        title_surface = self.title_font.render("HOCKEY ICE", True, ICE_BLUE)
        title_rect = title_surface.get_rect(center=(SCREEN_WIDTH//2, title_y))
        self.screen.blit(title_surface, title_rect)
        
        # Subtítulo con efecto de derretimiento
        subtitle_surface = self.subtitle_font.render("MELTING DOWN", True, FIRE_RED)
        subtitle_rect = subtitle_surface.get_rect(center=(SCREEN_WIDTH//2, title_y + 50))
        self.screen.blit(subtitle_surface, subtitle_rect)
        
        # Mensaje de misión
        mission_surface = self.text_font.render("ÚLTIMA ESPERANZA PARA SALVAR LA TIERRA", True, WARNING_ORANGE)
        mission_rect = mission_surface.get_rect(center=(SCREEN_WIDTH//2, title_y + 85))
        self.screen.blit(mission_surface, mission_rect)
        
        # Botón de jugar central
        self.play_button.draw(self.screen)
        
        # Botones circulares
        for button in self.circular_buttons:
            button.draw(self.screen)
        
        # Paneles informativos
        self.lore_panel.draw(self.screen)
        self.score_panel.draw(self.screen)
        
        # Mensaje ambiental rotativo
        message = self.environmental_messages[self.current_message]
        alpha = 128 + int(127 * math.sin(self.time * 0.1))
        
        # Fondo del mensaje
        message_surface = self.text_font.render(message, True, WHITE)
        message_rect = message_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT - 50))
        
        bg_rect = message_rect.inflate(20, 10)
        pygame.draw.rect(self.screen, (0, 0, 0, alpha//2), bg_rect)
        pygame.draw.rect(self.screen, WARNING_ORANGE, bg_rect, 2)
        
        self.screen.blit(message_surface, message_rect)
        
        # Indicador de estado planetario
        health_text = "ESTADO PLANETARIO: CRÍTICO"
        health_surface = pygame.font.Font(None, 24).render(health_text, True, FIRE_RED)
        health_rect = health_surface.get_rect(topright=(SCREEN_WIDTH - 20, 20))
        self.screen.blit(health_surface, health_rect)
    
    def draw(self):
        self.draw_main_interface()
    
    def run(self):
        print("🌍 Iniciando Hockey Ice Melting Down - Sistema GaiaCore activado")
        print("🏒 Prepárate para salvar el planeta, guardián...")
        
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = MainMenu()
    game.run()