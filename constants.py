# Constantes para el juego de Air Hockey

# Dimensiones
WIDTH, HEIGHT = 800, 500
FPS = 60

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
FRICTION = 0.999  # Fricción del puck en la superficie
MAX_PUCK_SPEED = 10  # Velocidad máxima del puck
COLLISION_ELASTICITY = 0.9  # Elasticidad en las colisiones (1.0 = perfectamente elástico)

# Parámetros del juego
GOAL_WIDTH_RATIO = 1/3  # La portería ocupa 1/3 de la altura