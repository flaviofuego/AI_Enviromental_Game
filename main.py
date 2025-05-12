import pygame
import sys
import os
import numpy as np
from constants import *
from sprites import Puck, HumanMallet, AIMallet
from table import Table
from dqn_agent import DQNAgent, RLAIMallet
from training import train_agent
from utils import draw_glow

def main(agent=None, use_rl=False):
    """Función principal del juego"""
    
    # Inicializar pygame
    pygame.init()
    
    # Crear pantalla
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Air Hockey - Advanced Physics")
    clock = pygame.time.Clock()
    
    # Crear objetos del juego
    table = Table()
    
    # Decidir qué tipo de IA usar
    if use_rl and agent:
        ai_mallet = RLAIMallet(agent)
    else:
        ai_mallet = AIMallet()
        
    human_mallet = HumanMallet()
    puck = Puck()
    
    # Sprites
    all_sprites = pygame.sprite.Group()
    all_sprites.add(human_mallet, ai_mallet, puck)
    
    # Marcador
    player_score = 0
    ai_score = 0
    font = pygame.font.Font(None, 36)
    
    # Bucle principal
    running = True
    show_fps = False
    
    while running:
        # Controlar eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    show_fps = not show_fps
                elif event.key == pygame.K_r:
                    # Reinicio completo del juego
                    # Reiniciar el puck con velocidad cero
                    puck.reset(zero_velocity=True)
                    puck.velocity = [0, 0]  # Cancelar completamente la velocidad
                    
                    # Reiniciar las posiciones de los mallets
                    human_mallet.position = [WIDTH // 4, HEIGHT // 2]
                    human_mallet.rect.center = human_mallet.position
                    
                    # Reiniciar el mallet de la IA según su tipo
                    if use_rl and agent:
                        ai_mallet.position = [WIDTH * 3 // 4, HEIGHT // 2]
                    else:
                        ai_mallet.position = [WIDTH * 3 // 4, HEIGHT // 2]
                    ai_mallet.rect.center = ai_mallet.position
                    
                    # Reiniciar velocidades de los mallets
                    human_mallet.velocity = [0, 0]
                    ai_mallet.velocity = [0, 0]
        
        # Actualizar
        mouse_pos = pygame.mouse.get_pos()
        human_mallet.update(mouse_pos)
        
        if use_rl and agent:
            ai_mallet.update(puck.position, puck, training=False)
        else:
            ai_mallet.update(puck.position)
            
        # Actualizar el disco
        puck.update()
        
        # Verificar colisiones con los mazos usando el nuevo método
        puck.check_mallet_collision(human_mallet)
        puck.check_mallet_collision(ai_mallet)
        
        # Verificar goles
        goal = table.is_goal(puck)
        
        if goal == "player":
            player_score += 1
            puck.reset("player")
        elif goal == "ai":
            ai_score += 1
            puck.reset("ai")
        
        # Dibujar
        table.draw(screen)
        
        # Dibujar efectos de brillo
        draw_glow(screen, (255, 0, 0), human_mallet.position, human_mallet.radius)
        draw_glow(screen, (0, 255, 0), ai_mallet.position, ai_mallet.radius)
        draw_glow(screen, (0, 0, 255), puck.position, puck.radius)
        
        # Dibujar sprites
        all_sprites.draw(screen)
        
        # Dibujar marcador
        score_text = font.render(f"{player_score} - {ai_score}", True, WHITE)
        screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))
        
        # Mostrar modo de juego
        mode_text = font.render("Modo: " + ("RL AI" if use_rl else "Simple AI"), True, WHITE)
        screen.blit(mode_text, (WIDTH // 2 - mode_text.get_width() // 2, HEIGHT - 30))
        
        # Mostrar controles
        controls_text = font.render("F: FPS | R: Reset", True, WHITE)
        screen.blit(controls_text, (10, HEIGHT - 30))
        
        # Mostrar FPS si está activado
        if show_fps:
            fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, WHITE)
            screen.blit(fps_text, (10, 10))
        
        # Actualizar pantalla
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    # Mostrar mensaje de bienvenida
    print("=== Air Hockey con Reinforcement Learning ===")
    print("Este es un proof of concept de un juego de Air Hockey")
    print("que puede usar aprendizaje por refuerzo para la IA.")
    print("\nOpciones:")
    print("1. Jugar contra IA simple")
    print("2. Entrenar modelo de RL (puede tomar tiempo)")
    print("3. Jugar contra IA entrenada (si existe el modelo)")
    
    choice = input("\nSeleccione una opción (1-3): ").strip()
    
    if choice == "1":
        main(use_rl=False)
    elif choice == "2":
        episodes = 500
        try:
            episodes_input = input("Número de episodios para entrenar (predeterminado: 500): ").strip()
            if episodes_input:
                episodes = int(episodes_input)
        except ValueError:
            print("Valor no válido, usando 500 episodios")
        
        # Crear agente
        state_size = 6
        action_size = 5
        agent = DQNAgent(state_size, action_size)
        
        # Entrenar agente
        agent = train_agent(agent, episodes=episodes)
        
        # Preguntar si quiere jugar después del entrenamiento
        play_after = input("¿Quieres jugar contra el modelo entrenado? (s/n): ").lower().startswith('s')
        if play_after:
            main(agent=agent, use_rl=True)
    elif choice == "3":
        model_path = "air_hockey_model.h5"
        
        if os.path.exists(model_path):
            # Cargar el modelo entrenado
            state_size = 6
            action_size = 5
            agent = DQNAgent(state_size, action_size)
            agent.load(model_path)
            agent.epsilon = 0.01  # Baja exploración en modo juego
            
            main(agent=agent, use_rl=True)
        else:
            print(f"No se encontró el modelo en {model_path}. Por favor, entrena el modelo primero.")
            choice = input("¿Quieres jugar contra la IA simple en su lugar? (s/n): ").lower().startswith('s')
            if choice:
                main(use_rl=False)
    else:
        print("Opción no válida, jugando con IA simple")
        main(use_rl=False)