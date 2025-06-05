import os
import pygame
# Constantes para el juego de Air Hockey

# Función para obtener dimensiones dinámicas que coincidan con el menú
def get_dynamic_dimensions():
    """Obtiene las dimensiones dinámicas basadas en el tamaño de pantalla, 
    coincidiendo con las del menú principal"""
    if not pygame.get_init():
        pygame.init()
    
    info = pygame.display.Info()
    screen_width = min(1200, info.current_w - 100)
    screen_height = min(800, info.current_h - 100)
    
    # Mantener proporción de aspecto similar al hockey (1.6:1)
    # Si la pantalla es muy ancha, limitamos por altura
    if screen_width / screen_height > 1.8:
        screen_width = int(screen_height * 1.6)
    # Si es muy alta, limitamos por ancho
    elif screen_height / screen_width > 0.8:
        screen_height = int(screen_width * 0.625)
    
    return screen_width, screen_height

# Dimensiones por defecto
DEFAULT_WIDTH, DEFAULT_HEIGHT = 800, 500

# Variables globales para dimensiones actuales
_current_width = DEFAULT_WIDTH
_current_height = DEFAULT_HEIGHT

def get_screen_dimensions():
    """Obtiene las dimensiones actuales de la pantalla"""
    return _current_width, _current_height

def set_screen_dimensions(width, height):
    """Establece las dimensiones de la pantalla y actualiza las constantes"""
    global _current_width, _current_height, WIDTH, HEIGHT
    _current_width, _current_height = width, height
    WIDTH, HEIGHT = width, height

def get_scale_factors():
    """Obtiene los factores de escala basados en las dimensiones actuales vs las por defecto"""
    scale_x = _current_width / DEFAULT_WIDTH
    scale_y = _current_height / DEFAULT_HEIGHT
    return scale_x, scale_y

# Dimensiones actuales (se pueden actualizar dinámicamente)
WIDTH, HEIGHT = DEFAULT_WIDTH, DEFAULT_HEIGHT
FPS = 120

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
NEON_RED = (255, 60, 60)
NEON_GREEN = (60, 255, 60)
NEON_BLUE = (60, 60, 255)

# Parámetros de física
FRICTION = 0.9991  # Fricción del puck en la superficie
MAX_PUCK_SPEED = 12  # Velocidad máxima del puck
COLLISION_ELASTICITY = 0.9  # Elasticidad en las colisiones (1.0 = perfectamente elástico)

# Parámetros del juego
GOAL_WIDTH_RATIO = 1/3  # La portería ocupa 1/3 de la altura
MULLET_RADIUS = 32

PATH = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(PATH, 'logs')
MODELS = os.path.join(PATH, 'models')