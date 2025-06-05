import pygame
import sys
import os

# Añadir el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.config.level_config import get_asset_path
from sprite_loader import SpriteLoader
from constants import set_screen_dimensions

def test_goal_scaling():
    """Prueba el escalado de porterías con la nueva relación de aspecto"""
    pygame.init()
    
    # Establecer dimensiones de pantalla para las pruebas
    set_screen_dimensions(1200, 800)
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Prueba de Escalado de Porterías")
    
    print("=== PRUEBA DE ESCALADO DE PORTERÍAS ===")
    print("Relación de aspecto base (Nivel 1): 520x949 = 0.548")
    
    # Simular zona de portería (como en el juego real)
    from constants import GOAL_WIDTH_RATIO
    current_height = 800
    goal_height = int(current_height * GOAL_WIDTH_RATIO)
    base_aspect_ratio = 520 / 949
    target_width = int(goal_height * base_aspect_ratio)
    
    print(f"Altura de portería en juego: {goal_height}")
    print(f"Ancho objetivo (con relación de aspecto): {target_width}")
    
    clock = pygame.time.Clock()
    running = True
    
    # Cargar porterías de todos los niveles
    goal_sprites = {}
    for level_id in range(1, 6):
        try:
            sprites = SpriteLoader.load_level_sprites(level_id)
            if 'goal_left' in sprites and 'goal_right' in sprites:
                goal_sprites[level_id] = {
                    'left': sprites['goal_left'],
                    'right': sprites['goal_right']
                }
                print(f"✓ Porterías cargadas para nivel {level_id}")
            else:
                print(f"✗ Porterías no encontradas para nivel {level_id}")
        except Exception as e:
            print(f"✗ Error cargando nivel {level_id}: {e}")
    
    current_level = 1
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    current_level = 1
                elif event.key == pygame.K_2:
                    current_level = 2
                elif event.key == pygame.K_3:
                    current_level = 3
                elif event.key == pygame.K_4:
                    current_level = 4
                elif event.key == pygame.K_5:
                    current_level = 5
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Limpiar pantalla
        screen.fill((50, 50, 50))
        
        # Dibujar líneas de referencia
        pygame.draw.line(screen, (255, 255, 255), (0, 100), (1200, 100), 2)  # Línea superior portería
        pygame.draw.line(screen, (255, 255, 255), (0, 100 + goal_height), (1200, 100 + goal_height), 2)  # Línea inferior portería
        pygame.draw.line(screen, (255, 255, 255), (600, 0), (600, 800), 2)  # Línea central
        
        # Mostrar información
        font = pygame.font.Font(None, 36)
        info_text = font.render(f"Nivel {current_level} - Presiona 1-5 para cambiar", True, (255, 255, 255))
        screen.blit(info_text, (10, 10))
        
        size_text = font.render(f"Tamaño objetivo: {target_width}x{goal_height}", True, (255, 255, 255))
        screen.blit(size_text, (10, 50))
        
        # Dibujar porterías del nivel actual
        if current_level in goal_sprites:
            # Portería izquierda
            left_goal = goal_sprites[current_level]['left']
            if left_goal:
                scaled_left = pygame.transform.smoothscale(left_goal, (target_width, goal_height))
                screen.blit(scaled_left, (0, 100))
                
                # Mostrar tamaño original
                orig_size = left_goal.get_size()
                orig_text = font.render(f"Original izq: {orig_size[0]}x{orig_size[1]}", True, (255, 255, 0))
                screen.blit(orig_text, (10, 100 + goal_height + 10))
            
            # Portería derecha
            right_goal = goal_sprites[current_level]['right']
            if right_goal:
                scaled_right = pygame.transform.smoothscale(right_goal, (target_width, goal_height))
                screen.blit(scaled_right, (1200 - target_width, 100))
                
                # Mostrar tamaño original
                orig_size = right_goal.get_size()
                orig_text = font.render(f"Original der: {orig_size[0]}x{orig_size[1]}", True, (255, 255, 0))
                screen.blit(orig_text, (1200 - 300, 100 + goal_height + 10))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    test_goal_scaling() 