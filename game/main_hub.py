import os
import sys
import pygame
import math
import random

# Añadir el directorio del proyecto al path de Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from game.pages.home import HockeyMainScreen
from game.pages.Level_Select import LevelSelectScreen
from game.config.save_system import GameSaveSystem
from game.components.AudioManager import audio_manager
from main_improved import main as start_game

def ice_melt_transition(screen, fade_out=True, duration=1.5):
    """Crea un efecto de transición temático con derretimiento de hielo"""
    width, height = screen.get_size()
    clock = pygame.time.Clock()
    
    # Colores temáticos
    ice_blue = (173, 216, 230)
    heat_red = (220, 50, 50)
    frost_white = (240, 248, 255)
    
    # Capturar la pantalla actual
    screenshot = screen.copy()
    
    # Crear partículas de hielo/calor
    particles = []
    for _ in range(150):
        particles.append({
            'x': random.randint(0, width),
            'y': random.randint(0, height),
            'size': random.randint(2, 6),
            'speed': random.uniform(1, 4),
            'angle': random.uniform(0, 2 * math.pi),
            'color': random.choice([ice_blue, frost_white, (100, 150, 200)]),
            'life': random.uniform(0.5, 1.0)
        })
    
    # Duración total en frames (60 FPS)
    total_frames = int(duration * 60)
    
    for frame in range(total_frames):
        dt = clock.tick(60) / 1000.0
        progress = frame / total_frames
        
        # Limpiar pantalla con gradiente
        if fade_out:
            # De imagen a efecto de hielo
            alpha = int(255 * (1 - progress))
            effect_alpha = int(255 * progress)
        else:
            # De efecto de hielo a imagen
            alpha = int(255 * progress)
            effect_alpha = int(255 * (1 - progress))
        
        # Dibujar screenshot con transparencia
        temp_surface = screenshot.copy()
        temp_surface.set_alpha(alpha)
        screen.blit(temp_surface, (0, 0))
        
        # Crear superficie de efecto
        effect_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Dibujar gradiente de fondo para el efecto
        for y in range(0, height, 5):  # Saltar líneas para rendimiento
            ratio = y / height
            if fade_out:
                # Transición de azul hielo a rojo calor
                color_r = int(ice_blue[0] * (1 - progress) + heat_red[0] * progress)
                color_g = int(ice_blue[1] * (1 - progress) + heat_red[1] * progress)
                color_b = int(ice_blue[2] * (1 - progress) + heat_red[2] * progress)
            else:
                # Transición de rojo calor a azul hielo
                color_r = int(heat_red[0] * (1 - progress) + ice_blue[0] * progress)
                color_g = int(heat_red[1] * (1 - progress) + ice_blue[1] * progress)
                color_b = int(heat_red[2] * (1 - progress) + ice_blue[2] * progress)
            
            # Añadir variación vertical
            color_r = int(color_r * (1 - ratio * 0.3))
            color_g = int(color_g * (1 - ratio * 0.3))
            color_b = int(color_b * (1 - ratio * 0.3))
            
            pygame.draw.line(effect_surface, (color_r, color_g, color_b, effect_alpha), 
                           (0, y), (width, y))
        
        # Actualizar y dibujar partículas
        for particle in particles:
            # Movimiento
            particle['x'] += math.cos(particle['angle']) * particle['speed']
            particle['y'] += math.sin(particle['angle']) * particle['speed'] * 0.5
            
            # Efecto de derretimiento (las partículas caen)
            if fade_out:
                particle['y'] += progress * 3
            
            # Vida de la partícula
            particle['life'] -= dt * 0.5
            
            # Redibujar si sale de pantalla
            if particle['y'] > height or particle['life'] <= 0:
                particle['y'] = -10
                particle['x'] = random.randint(0, width)
                particle['life'] = random.uniform(0.5, 1.0)
            
            # Dibujar partícula con efecto de brillo
            particle_alpha = int(particle['life'] * effect_alpha)
            if particle_alpha > 0:
                # Núcleo de la partícula
                pygame.draw.circle(effect_surface, 
                                 (*particle['color'], particle_alpha),
                                 (int(particle['x']), int(particle['y'])),
                                 particle['size'])
                
                # Halo brillante
                halo_size = particle['size'] + 2
                halo_alpha = int(particle_alpha * 0.3)
                pygame.draw.circle(effect_surface,
                                 (*frost_white, halo_alpha),
                                 (int(particle['x']), int(particle['y'])),
                                 halo_size, 1)
        
        # Efecto de onda de calor/frío
        if progress > 0.3 and progress < 0.7:
            wave_intensity = math.sin((progress - 0.3) * math.pi / 0.4)
            for x in range(0, width, 20):
                wave_y = height // 2 + math.sin(x * 0.02 + frame * 0.1) * 50 * wave_intensity
                pygame.draw.circle(effect_surface,
                                 (*frost_white, int(30 * wave_intensity)),
                                 (x, int(wave_y)), 15)
        
        # Aplicar superficie de efecto
        screen.blit(effect_surface, (0, 0))
        
        # Efecto de cristalización en los bordes
        if fade_out and progress > 0.5:
            crystal_alpha = int((progress - 0.5) * 2 * 100)
            # Bordes superiores e inferiores
            for i in range(int(progress * 50)):
                pygame.draw.line(screen, (*frost_white, crystal_alpha),
                               (0, i), (width, i))
                pygame.draw.line(screen, (*frost_white, crystal_alpha),
                               (0, height - i), (width, height - i))
        
        pygame.display.flip()
    
    # Asegurar que la pantalla quede limpia al final
    if not fade_out:
        screen.blit(screenshot, (0, 0))
        pygame.display.flip()

def transition_effect(screen, fade_out=True):
    """Wrapper para mantener compatibilidad pero usar la nueva transición"""
    ice_melt_transition(screen, fade_out, duration=0.5)

# Ejecutar el bucle principal del juego
if __name__ == "__main__":
    # Inicializar pygame una sola vez
    pygame.init()
    
    # Crear ventana principal única
    info = pygame.display.Info()
    screen_width = min(1200, info.current_w - 100)
    screen_height = min(800, info.current_h - 100)
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Hockey Is Melting Down - Salva la Tierra")
    
    # Estado inicial del juego
    current_screen = "home"
    save_system = GameSaveSystem()  # Inicializar sistema de guardado una sola vez
    
    # Cargar el último perfil usado automáticamente
    last_profile = save_system.get_last_used_profile()
    if last_profile:
        print(f"Cargando último perfil: {last_profile['player_name']}")
    
    selected_level = None
    exit_game = False
    
    # Iniciar música del menú principal
    audio_manager.preload_audio_for_screen("home")
    audio_manager.play_music("home")
    
    # Bucle principal del controlador de pantallas
    while not exit_game:
        if current_screen == "home":
            # Asegurar que esté sonando la música del home
            audio_manager.play_music("home")
            
            # Iniciar pantalla principal con la ventana existente y el sistema de guardado
            main_screen = HockeyMainScreen(screen, save_system)
            
            # Efecto de aparición si venimos de otra pantalla
            transition_effect(screen, fade_out=False)
            
            # Ejecutar pantalla principal y obtener siguiente acción
            result = main_screen.run()
            
            # Procesar resultado de la pantalla principal
            if result == "exit":
                exit_game = True
            elif result == "level_select":
                audio_manager.play_sound_effect("transition")
                transition_effect(screen, fade_out=True)
                current_screen = "level_select"
            
        elif current_screen == "level_select":
            # Cambiar música para selección de niveles
            audio_manager.play_music("level_select")
            audio_manager.preload_audio_for_screen("level_select")
            
            # Iniciar pantalla de selección de nivel con la ventana existente y el sistema de guardado
            level_screen = LevelSelectScreen(save_system, screen)
            
            # Efecto de aparición
            transition_effect(screen, fade_out=False)
            
            # Ejecutar pantalla de selección y obtener resultado
            result = level_screen.run()
            
            # Procesar resultado de la selección de niveles
            if result == "exit":
                exit_game = True
            elif result == "back_to_menu":
                audio_manager.play_sound_effect("transition")
                transition_effect(screen, fade_out=True)
                current_screen = "home"
            elif result and result.startswith("start_level_"):
                # Cambiar a música de gameplay
                audio_manager.play_music("gameplay")
                audio_manager.preload_audio_for_screen("gameplay")
                
                # Aquí implementarías la carga del nivel específico
                # Extract level ID and start the game
                level_id = int(result.split("_")[-1])
                print(f"Iniciando nivel {level_id}")
                
                # Start the game with the selected level
                transition_effect(screen, fade_out=True)
                start_game(level_id=level_id, debug_mode=False)
                
                # Return to menu after game ends
                transition_effect(screen, fade_out=True)
                current_screen = "home"
    
    # Limpiar recursos de audio al salir
    audio_manager.cleanup()
    
    # Finalizar Pygame al salir
    pygame.quit()
    sys.exit()