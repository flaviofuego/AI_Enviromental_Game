import pygame
import random
import math
import numpy as np
from constants import *
from sprites import Puck, HumanMallet
from dqn_agent import RLAIMallet
from table import Table
from utils import vector_length

def train_agent(agent, episodes=500, batch_size=32, model_name="air_hockey_model.h5"):
    """Entrena al agente de RL"""
    
    print("Iniciando entrenamiento del agente RL...")
    
    # Crear objetos
    table = Table()
    puck = Puck()
    
    ai_mallet = RLAIMallet(agent)
    human_mallet = HumanMallet()  # Para autojuego, se movería siguiendo una estrategia simple
    
    # Para colisiones
    all_sprites = pygame.sprite.Group()
    all_sprites.add(human_mallet, ai_mallet, puck)
    
    # Preparar pantalla
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Air Hockey RL Training")
    clock = pygame.time.Clock()
    
    # Bucle de entrenamiento
    for episode in range(episodes):
        # Reiniciar el juego
        puck.reset()
        ai_mallet.position = [WIDTH * 3 // 4, HEIGHT // 2]
        ai_mallet.rect.center = ai_mallet.position
        human_mallet.position = [WIDTH // 4, HEIGHT // 2]
        human_mallet.rect.center = human_mallet.position
        
        # Variables para seguimiento
        previous_distance = math.sqrt((puck.position[0] - ai_mallet.position[0])**2 + 
                                   (puck.position[1] - ai_mallet.position[1])**2)
        player_score = 0
        ai_score = 0
        
        done = False
        steps = 0
        
        while not done and steps < 1000:  # Limitamos los pasos para evitar episodios infinitos
            steps += 1
            
            # Procesar eventos para poder cerrar la ventana durante el entrenamiento
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return agent
            
            # Actualizar jugador simulado (por simplicidad, sigue la bola con ruido)
            if puck.position[0] < WIDTH // 2:
                target_y = puck.position[1] + random.randint(-30, 30)
                human_mallet.position[1] += (target_y - human_mallet.position[1]) * 0.1
                human_mallet.rect.center = human_mallet.position
            
            # Obtener estado y acción para la IA
            action = ai_mallet.update(puck.position, puck, training=True)
            
            # Actualizar el disco
            old_puck_pos = puck.position.copy()
            puck.update()
            
            # Verificar colisiones con los mazos usando el nuevo método
            ai_hit_puck = puck.check_mallet_collision(ai_mallet)
            human_hit_puck = puck.check_mallet_collision(human_mallet)
            
            # Verificar goles
            goal = table.is_goal(puck)
            goal_scored = False
            
            if goal == "player":
                player_score += 1
                goal_scored = True
                puck.reset("player")
            elif goal == "ai":
                ai_score += 1
                goal_scored = True
                puck.reset("ai")
            
            # Calcular recompensas
            reward = 0
            
            # Recompensa por acercarse al disco
            current_distance = math.sqrt((puck.position[0] - ai_mallet.position[0])**2 + 
                                      (puck.position[1] - ai_mallet.position[1])**2)
            
            if current_distance < previous_distance and puck.position[0] > WIDTH // 2:
                reward += 0.1
            previous_distance = current_distance
            
            # Recompensa por golpear el disco
            if ai_hit_puck:
                reward += 1.0
                
                # Recompensa adicional si el golpe va hacia la portería del oponente
                if puck.velocity[0] < 0:
                    reward += 0.5
            
            # Grandes recompensas/penalizaciones por goles
            if goal_scored:
                if goal == "player":
                    # Jugador anotó
                    reward -= 10.0
                else:
                    # IA anotó
                    reward += 10.0
                
                done = player_score >= 5 or ai_score >= 5
            
            # Guardar experiencia y entrenar
            next_state = ai_mallet.get_state(puck)
            agent.remember(ai_mallet.state, action, reward, next_state, done)
            
            # Entrenar el modelo
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
            # Visualización del entrenamiento (opcional pero útil)
            if episode % 10 == 0:  # Mostrar cada 10 episodios para no ralentizar
                table.draw(screen)
                all_sprites.draw(screen)
                
                # Mostrar marcador
                font = pygame.font.Font(None, 36)
                score_text = font.render(f"{player_score} - {ai_score} (Ep: {episode})", True, WHITE)
                screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))
                
                pygame.display.flip()
                clock.tick(120)  # Entrenamiento acelerado
                
        # Mostrar progreso
        if episode % 10 == 0:
            print(f"Episodio {episode}/{episodes}, Epsilon: {agent.epsilon:.2f}, Marcador: IA {ai_score} - Jugador {player_score}")
            
    # Guardar el modelo entrenado
    agent.save(model_name)
    print(f"Entrenamiento completado. Modelo guardado como {model_name}")
    return agent