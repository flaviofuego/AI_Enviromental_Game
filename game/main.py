import os
import sys
import pygame

# Añadir el directorio del proyecto al path de Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from game.pages.home import HockeyMainScreen
from game.pages.Level_Select import LevelSelectScreen

def transition_effect(screen, fade_out=True):
    """Crea un efecto de transición suave entre pantallas"""
    width, height = screen.get_size()
    fade_surface = pygame.Surface((width, height))
    fade_surface.fill((0, 0, 0))  # Superficie negra para desvanecer
    
    # 30 pasos de desvanecimiento (ajustar para velocidad deseada)
    for alpha in range(0, 256, 8):
        if fade_out:
            # Desvanecer a negro
            fade_surface.set_alpha(alpha)
        else:
            # Desvanecer desde negro
            fade_surface.set_alpha(255 - alpha)
        
        screen.blit(fade_surface, (0, 0))
        pygame.display.flip()
        pygame.time.delay(10)  # Pequeña pausa para visualizar la transición

# Ejecutar el bucle principal del juego
if __name__ == "__main__":
    pygame.init()
    
    # Estado inicial del juego
    current_screen = "home"
    save_system = None  # Sistema de guardado compartido entre pantallas
    selected_level = None
    exit_game = False
    
    # Bucle principal del controlador de pantallas
    while not exit_game:
        if current_screen == "home":
            # Iniciar pantalla principal
            main_screen = HockeyMainScreen()
            # Obtener superficie de pantalla para transiciones
            screen = main_screen.screen
            
            # Efecto de aparición si venimos de otra pantalla
            transition_effect(screen, fade_out=False)
            
            # Ejecutar pantalla principal y obtener siguiente acción
            result = main_screen.run()
            save_system = main_screen.save_system  # Guardar referencia al sistema de guardado
            
            # Procesar resultado de la pantalla principal
            if result == "exit":
                exit_game = True
            elif result == "level_select":
                transition_effect(screen, fade_out=True)
                current_screen = "level_select"
            
        elif current_screen == "level_select":
            # Iniciar pantalla de selección de nivel con el sistema de guardado compartido
            level_screen = LevelSelectScreen(save_system)
            screen = level_screen.screen
            
            # Efecto de aparición
            transition_effect(screen, fade_out=False)
            
            # Ejecutar pantalla de selección y obtener resultado
            result = level_screen.run()
            
            # Procesar resultado de la selección de niveles
            if result == "back_to_menu":
                transition_effect(screen, fade_out=True)
                current_screen = "home"
            elif result and result.startswith("start_level_"):
                # Aquí implementarías la carga del nivel específico
                level_id = int(result.split("_")[-1])
                print(f"Iniciando nivel {level_id}")
                # Por ahora, volvemos al menú principal
                transition_effect(screen, fade_out=True)
                current_screen = "home"
    
    # Finalizar Pygame al salir
    pygame.quit()
    sys.exit()