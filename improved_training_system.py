# improved_training_system.py
import gymnasium as gym
import numpy as np
import time
import os
import pygame
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
import matplotlib.pyplot as plt
from collections import deque

torch.set_num_threads(6)
torch.cuda.is_available = lambda: False

class AdvancedRewardCallback(BaseCallback):
    """Callback para monitorear y ajustar el sistema de recompensas"""
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.goal_ratios = deque(maxlen=100)
        
    def _on_step(self):
        # Recopilar estadísticas del episodio
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Calcular ratio de goles
                ai_score = info.get('ai_score', 0)
                player_score = info.get('player_score', 0)
                total_goals = ai_score + player_score
                goal_ratio = ai_score / max(1, total_goals)
                self.goal_ratios.append(goal_ratio)
                
                # Log estadísticas cada 100 episodios
                if len(self.episode_rewards) == 100:
                    avg_reward = np.mean(self.episode_rewards)
                    avg_length = np.mean(self.episode_lengths)
                    avg_goal_ratio = np.mean(self.goal_ratios)
                    
                    if self.verbose > 0:
                        print(f"Step {self.n_calls}: Avg Reward: {avg_reward:.2f}, "
                              f"Avg Length: {avg_length:.1f}, Goal Ratio: {avg_goal_ratio:.2f}")
        
        return True

class CurriculumLearningCallback(BaseCallback):
    """Callback para implementar curriculum learning progresivo"""
    def __init__(self, eval_env, eval_freq=50000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.performance_history = deque(maxlen=10)
        self.difficulty_level = 0
        self.min_performance_threshold = 0.6  # 60% win rate para avanzar
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluar rendimiento actual
            performance = self._evaluate_performance()
            self.performance_history.append(performance)
            
            # Decidir si aumentar dificultad
            if len(self.performance_history) >= 3:
                recent_performance = np.mean(list(self.performance_history)[-3:])
                
                if recent_performance > self.min_performance_threshold:
                    self._increase_difficulty()
                elif recent_performance < 0.3:  # Si está muy mal, reducir dificultad
                    self._decrease_difficulty()
            
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Performance: {performance:.2f}, "
                      f"Difficulty Level: {self.difficulty_level}")
        
        return True
    
    def _evaluate_performance(self):
        """Evalúa el rendimiento del agente"""
        wins = 0
        total_episodes = 5
        
        for _ in range(total_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                done = done or truncated
            
            # Contar victoria si AI tiene más goles
            if info.get('ai_score', 0) > info.get('player_score', 0):
                wins += 1
        
        return wins / total_episodes
    
    def _increase_difficulty(self):
        """Aumenta la dificultad del oponente"""
        if self.difficulty_level < 5:
            self.difficulty_level += 1
            unwrapped_env = self.eval_env.unwrapped
            unwrapped_env.set_difficulty_level(self.difficulty_level)
            
            # Ajustar threshold para el siguiente nivel
            self.min_performance_threshold = min(0.8, 0.5 + 0.1 * self.difficulty_level)
            
            if self.verbose > 0:
                print(f"Increased difficulty to level {self.difficulty_level}")
    
    def _decrease_difficulty(self):
        """Reduce la dificultad si el agente está luchando"""
        if self.difficulty_level > 0:
            self.difficulty_level -= 1
            unwrapped_env = self.eval_env.unwrapped
            unwrapped_env.set_difficulty_level(self.difficulty_level)
            
            if self.verbose > 0:
                print(f"Decreased difficulty to level {self.difficulty_level}")

class ImprovedAirHockeyEnv(gym.Env):
    """Entorno mejorado de Air Hockey con mejor sistema de recompensas y oponente inteligente"""
    
    def __init__(self, render_mode=None, play_mode=False):
        super().__init__()
        
        # Importar constantes y clases necesarias
        from constants import WIDTH, HEIGHT, WHITE
        from sprites import Puck, HumanMallet
        from table import Table
        
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.WHITE = WHITE
        self.play_mode = play_mode
        
        # Espacios de acción y observación
        self.action_space = gym.spaces.Discrete(5)  # Up, Down, Left, Right, Stay
        
        # Observación expandida (21 dimensiones)
        low = np.array([-1] * 21, dtype=np.float32)
        high = np.array([1] * 21, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        
        # Configuración de renderizado
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Objetos del juego
        self.table = Table()
        self.puck = None
        self.human_mallet = None
        self.ai_mallet = None
        
        # Estado del juego
        self.player_score = 0
        self.ai_score = 0
        self.steps = 0
        self.max_steps = 2000  # Episodios más largos
        
        # Sistema de dificultad mejorado
        self.difficulty_level = 0
        self.opponent_configs = {
            0: {"skill": 0.2, "speed": 0.3, "prediction": 0.1, "aggression": 0.2},
            1: {"skill": 0.35, "speed": 0.45, "prediction": 0.25, "aggression": 0.3},
            2: {"skill": 0.5, "speed": 0.6, "prediction": 0.4, "aggression": 0.45},
            3: {"skill": 0.65, "speed": 0.75, "prediction": 0.6, "aggression": 0.6},
            4: {"skill": 0.8, "speed": 0.9, "prediction": 0.8, "aggression": 0.75},
            5: {"skill": 0.95, "speed": 1.0, "prediction": 0.95, "aggression": 0.9}
        }
        
        # Métricas para recompensas
        self.last_puck_distance = 0
        self.consecutive_hits = 0
        self.time_since_last_hit = 0
        self.defensive_position_bonus = 0
        self.puck_control_time = 0
        
        # Historia para análisis
        self.episode_stats = {
            'hits': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'avg_distance_to_puck': 0,
            'defensive_actions': 0
        }
        
        self.reset()
    
    def set_difficulty_level(self, level):
        """Establece el nivel de dificultad del oponente"""
        self.difficulty_level = max(0, min(5, level))
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Importar clases necesarias
        from sprites import Puck, HumanMallet
        
        # Reiniciar objetos del juego
        self.puck = Puck()
        self.human_mallet = HumanMallet()
        
        # Configurar AI mallet
        self.ai_mallet_radius = 30
        self.ai_mallet_position = [self.WIDTH * 3 // 4, self.HEIGHT // 2]
        self.ai_mallet_velocity = [0, 0]
        
        # Crear sprite para visualización
        self.ai_mallet = pygame.sprite.Sprite()
        self.ai_mallet.image = pygame.Surface((self.ai_mallet_radius * 2, self.ai_mallet_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.ai_mallet.image, (0, 255, 0), (self.ai_mallet_radius, self.ai_mallet_radius), self.ai_mallet_radius)
        self.ai_mallet.rect = self.ai_mallet.image.get_rect(center=self.ai_mallet_position)
        
        # Reiniciar estado del juego
        self.player_score = 0
        self.ai_score = 0
        self.steps = 0
        
        # Reiniciar métricas
        self.last_puck_distance = self._calculate_distance_to_puck()
        self.consecutive_hits = 0
        self.time_since_last_hit = 0
        self.puck_control_time = 0
        
        # Reiniciar estadísticas del episodio
        self.episode_stats = {
            'hits': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'avg_distance_to_puck': 0,
            'defensive_actions': 0
        }
        
        # Actualizar oponente humano
        if not self.play_mode:
            self._update_intelligent_opponent()
        
        return self._get_enhanced_observation(), {}
    
    def step(self, action):
        self.steps += 1
        self.time_since_last_hit += 1
        
        # Aplicar acción del AI
        prev_position = self.ai_mallet_position.copy()
        self._apply_action(action)
        
        # Actualizar oponente humano (solo en modo entrenamiento)
        if not self.play_mode:
            self._update_intelligent_opponent()
        
        # Guardar distancia previa para cálculo de recompensa
        prev_distance = self._calculate_distance_to_puck()
        
        # Actualizar física del puck
        self.puck.update()
        
        # Verificar colisiones
        ai_hit = self._check_ai_collision()
        human_hit = self.puck.check_mallet_collision(self.human_mallet)
        
        # Actualizar métricas
        if ai_hit:
            self.consecutive_hits += 1
            self.time_since_last_hit = 0
            self.episode_stats['hits'] += 1
        
        # Verificar goles
        goal = self.table.is_goal(self.puck)
        goal_scored = False
        
        if goal == "player":
            self.player_score += 1
            self.episode_stats['goals_conceded'] += 1
            goal_scored = True
            self.puck.reset("player")
        elif goal == "ai":
            self.ai_score += 1
            self.episode_stats['goals_scored'] += 1
            goal_scored = True
            self.puck.reset("ai")
        
        # Calcular recompensa mejorada
        reward = self._calculate_advanced_reward(prev_distance, ai_hit, goal, action)
        
        # Verificar si el episodio terminó
        done = (goal_scored or self.steps >= self.max_steps or 
                self.player_score >= 7 or self.ai_score >= 7)
        
        truncated = self.steps >= self.max_steps and not goal_scored
        
        # Obtener observación mejorada
        observation = self._get_enhanced_observation()
        
        # Información adicional
        info = {
            "player_score": self.player_score,
            "ai_score": self.ai_score,
            "steps": self.steps,
            "hit_puck": ai_hit,
            "episode_stats": self.episode_stats.copy() if done else {}
        }
        
        return observation, reward, done, truncated, info
    
    def _apply_action(self, action):
        """Aplica la acción del AI con movimiento más fluido"""
        move_amount = 8  # Movimiento más rápido
        
        if action == 0:  # Up
            self.ai_mallet_position[1] = max(self.ai_mallet_position[1] - move_amount, self.ai_mallet_radius)
        elif action == 1:  # Down
            self.ai_mallet_position[1] = min(self.ai_mallet_position[1] + move_amount, self.HEIGHT - self.ai_mallet_radius)
        elif action == 2:  # Left
            self.ai_mallet_position[0] = max(self.ai_mallet_position[0] - move_amount, self.WIDTH // 2 + self.ai_mallet_radius)
        elif action == 3:  # Right
            self.ai_mallet_position[0] = min(self.ai_mallet_position[0] + move_amount, self.WIDTH - self.ai_mallet_radius)
        # action == 4 es "stay" (no hacer nada)
        
        # Actualizar sprite y velocidad
        self.ai_mallet.rect.center = self.ai_mallet_position
        self.ai_mallet_velocity = [
            self.ai_mallet_position[0] - self.ai_mallet.rect.centerx,
            self.ai_mallet_position[1] - self.ai_mallet.rect.centery
        ]
    
    def _update_intelligent_opponent(self):
        """Oponente humano inteligente con diferentes niveles de dificultad"""
        config = self.opponent_configs[self.difficulty_level]
        
        # Parámetros del oponente basados en dificultad
        skill = config["skill"]
        speed = config["speed"]
        prediction_ability = config["prediction"]
        aggression = config["aggression"]
        
        # Calcular posición objetivo basada en estrategia
        target_x, target_y = self._calculate_opponent_strategy(skill, prediction_ability, aggression)
        
        # Aplicar movimiento con velocidad limitada
        max_speed = 12 * speed
        
        dx = target_x - self.human_mallet.position[0]
        dy = target_y - self.human_mallet.position[1]
        
        distance = np.sqrt(dx**2 + dy**2)
        if distance > max_speed:
            dx = (dx / distance) * max_speed
            dy = (dy / distance) * max_speed
        
        # Actualizar posición del oponente
        new_x = np.clip(self.human_mallet.position[0] + dx, 
                       self.human_mallet.radius, 
                       self.WIDTH // 2 - self.human_mallet.radius)
        new_y = np.clip(self.human_mallet.position[1] + dy, 
                       self.human_mallet.radius, 
                       self.HEIGHT - self.human_mallet.radius)
        
        self.human_mallet.position = [new_x, new_y]
        self.human_mallet.rect.center = self.human_mallet.position
        self.human_mallet.velocity = [dx, dy]
    
    def _calculate_opponent_strategy(self, skill, prediction_ability, aggression):
        """Calcula la estrategia del oponente basada en el estado del juego"""
        puck_x, puck_y = self.puck.position
        puck_vx, puck_vy = self.puck.velocity
        
        # Predicción de trayectoria del puck
        if abs(puck_vx) > 0.1:
            time_to_reach = (self.human_mallet.position[0] - puck_x) / puck_vx
            predicted_y = puck_y + puck_vy * time_to_reach * prediction_ability
        else:
            predicted_y = puck_y
        
        # Estrategia basada en posición del puck
        if puck_x < self.WIDTH // 2:  # Puck en lado del jugador humano
            if np.sqrt((puck_x - self.human_mallet.position[0])**2 + 
                      (puck_y - self.human_mallet.position[1])**2) < 100 * aggression:
                # Modo agresivo: ir por el puck
                target_x = min(puck_x + 20, self.WIDTH // 2 - self.human_mallet.radius)
                target_y = predicted_y
                
                # Si está cerca, apuntar hacia la portería del AI
                if np.sqrt((puck_x - self.human_mallet.position[0])**2 + 
                          (puck_y - self.human_mallet.position[1])**2) < 50:
                    # Calcular ángulo hacia la portería
                    goal_center_y = self.HEIGHT // 2
                    angle_to_goal = np.arctan2(goal_center_y - puck_y, self.WIDTH - puck_x)
                    
                    # Posicionarse para golpear hacia la portería
                    offset_distance = 40
                    target_x = puck_x - offset_distance * np.cos(angle_to_goal) * skill
                    target_y = puck_y - offset_distance * np.sin(angle_to_goal) * skill
            else:
                # Modo defensivo: posicionarse estratégicamente
                target_x = self.WIDTH * 0.2
                target_y = self.HEIGHT // 2 + (predicted_y - self.HEIGHT // 2) * 0.7
        else:  # Puck en lado del AI
            # Modo defensivo: proteger portería
            target_x = self.WIDTH * 0.15
            
            # Anticipar devolución del puck
            if puck_vx < 0:  # Puck viene hacia el jugador
                target_y = predicted_y
            else:
                target_y = self.HEIGHT // 2
        
        # Añadir algo de ruido para hacer el comportamiento menos predecible
        noise_factor = (1.0 - skill) * 20
        target_x += np.random.normal(0, noise_factor)
        target_y += np.random.normal(0, noise_factor)
        
        # Asegurar que el objetivo esté dentro de los límites
        target_x = np.clip(target_x, self.human_mallet.radius, self.WIDTH // 2 - self.human_mallet.radius)
        target_y = np.clip(target_y, self.human_mallet.radius, self.HEIGHT - self.human_mallet.radius)
        
        return target_x, target_y
    
    def _calculate_advanced_reward(self, prev_distance, ai_hit, goal, action):
        """Sistema de recompensas avanzado y balanceado"""
        reward = 0.0
        
        # 1. Recompensas por goles (más importantes)
        if goal == "ai":
            reward += 100.0  # Gran recompensa por anotar
        elif goal == "player":
            reward -= 50.0   # Penalización por recibir gol
        
        # 2. Recompensas por golpear el puck
        if ai_hit:
            base_hit_reward = 10.0
            
            # Bonus por hits consecutivos (hasta un límite)
            consecutive_bonus = min(self.consecutive_hits * 2.0, 10.0)
            
            # Bonus por velocidad del golpe
            puck_speed = np.sqrt(self.puck.velocity[0]**2 + self.puck.velocity[1]**2)
            speed_bonus = min(puck_speed * 0.5, 5.0)
            
            # Bonus por dirección hacia la portería del oponente
            direction_to_goal = np.array([0 - self.puck.position[0], self.HEIGHT//2 - self.puck.position[1]])
            direction_to_goal = direction_to_goal / (np.linalg.norm(direction_to_goal) + 1e-8)
            
            puck_direction = np.array(self.puck.velocity)
            puck_direction = puck_direction / (np.linalg.norm(puck_direction) + 1e-8)
            
            direction_alignment = np.dot(direction_to_goal, puck_direction)
            direction_bonus = max(0, direction_alignment * 5.0)
            
            reward += base_hit_reward + consecutive_bonus + speed_bonus + direction_bonus
            
        else:
            # Resetear hits consecutivos si no golpeó
            self.consecutive_hits = 0
        
        # 3. Recompensas por posicionamiento estratégico
        current_distance = self._calculate_distance_to_puck()
        
        # Recompensar acercarse al puck cuando está en el lado del AI
        if self.puck.position[0] > self.WIDTH // 2:
            if current_distance < prev_distance:
                reward += 0.5  # Recompensa por acercarse
            else:
                reward -= 0.2  # Pequeña penalización por alejarse
        
        # 4. Recompensas por posición defensiva
        if self.puck.position[0] < self.WIDTH // 2:  # Puck en lado del oponente
            # Posición defensiva ideal
            ideal_defensive_x = self.WIDTH * 0.75
            ideal_defensive_y = self.puck.position[1]  # Alinearse con el puck
            
            defensive_distance = np.sqrt(
                (self.ai_mallet_position[0] - ideal_defensive_x)**2 + 
                (self.ai_mallet_position[1] - ideal_defensive_y)**2
            )
            
            # Recompensar buena posición defensiva
            max_defensive_distance = self.WIDTH * 0.3
            defensive_reward = max(0, 1.0 - defensive_distance / max_defensive_distance)
            reward += defensive_reward * 0.3
        
        # 5. Penalizaciones por comportamiento ineficiente
        
        # Penalización por movimiento excesivo sin propósito
        if action != 4:  # Si no es "stay"
            movement_penalty = 0.05
            reward -= movement_penalty
        
        # Penalización por tiempo sin tocar el puck
        if self.time_since_last_hit > 100:
            timeout_penalty = min((self.time_since_last_hit - 100) * 0.01, 2.0)
            reward -= timeout_penalty
        
        # 6. Bonus por control del juego
        
        # Bonus si el puck se mueve hacia la portería del oponente
        if self.puck.velocity[0] < -1.0:  # Moviéndose hacia la izquierda (portería del oponente)
            reward += 0.2
        
        # Bonus por mantener el puck en el lado del oponente
        if self.puck.position[0] < self.WIDTH // 2:
            reward += 0.1
        
        # 7. Recompensas adaptativas basadas en la dificultad
        difficulty_multiplier = 1.0 + (self.difficulty_level * 0.1)
        reward *= difficulty_multiplier
        
        return reward
    
    def _calculate_distance_to_puck(self):
        """Calcula la distancia del AI mallet al puck"""
        return np.sqrt(
            (self.puck.position[0] - self.ai_mallet_position[0])**2 + 
            (self.puck.position[1] - self.ai_mallet_position[1])**2
        )
    
    def _get_enhanced_observation(self):
        """Observación mejorada con más información contextual"""
        # Normalizar posiciones
        ai_x_norm = self.ai_mallet_position[0] / self.WIDTH
        ai_y_norm = self.ai_mallet_position[1] / self.HEIGHT
        puck_x_norm = self.puck.position[0] / self.WIDTH
        puck_y_norm = self.puck.position[1] / self.HEIGHT
        human_x_norm = self.human_mallet.position[0] / self.WIDTH
        human_y_norm = self.human_mallet.position[1] / self.HEIGHT
        
        # Normalizar velocidades
        max_velocity = 20.0
        puck_vx_norm = np.clip(self.puck.velocity[0] / max_velocity, -1, 1)
        puck_vy_norm = np.clip(self.puck.velocity[1] / max_velocity, -1, 1)
        ai_vx_norm = np.clip(self.ai_mallet_velocity[0] / max_velocity, -1, 1)
        ai_vy_norm = np.clip(self.ai_mallet_velocity[1] / max_velocity, -1, 1)
        human_vx_norm = np.clip(self.human_mallet.velocity[0] / max_velocity, -1, 1)
        human_vy_norm = np.clip(self.human_mallet.velocity[1] / max_velocity, -1, 1)
        
        # Distancias normalizadas
        max_distance = np.sqrt(self.WIDTH**2 + self.HEIGHT**2)
        puck_to_ai_dist = self._calculate_distance_to_puck() / max_distance
        puck_to_human_dist = np.sqrt(
            (self.puck.position[0] - self.human_mallet.position[0])**2 + 
            (self.puck.position[1] - self.human_mallet.position[1])**2
        ) / max_distance
        
        # Información contextual
        puck_in_ai_half = 1.0 if self.puck.position[0] > self.WIDTH // 2 else -1.0
        puck_moving_to_ai_goal = 1.0 if self.puck.velocity[0] > 0 else -1.0
        puck_moving_to_human_goal = 1.0 if self.puck.velocity[0] < 0 else -1.0
        
        # Información de tiempo y estado
        time_factor = min(self.time_since_last_hit / 100.0, 1.0)
        score_diff = (self.ai_score - self.player_score) / 7.0  # Normalizado por puntuación máxima
        
        # Predicción de trayectoria del puck
        if abs(self.puck.velocity[0]) > 0.1:
            time_to_ai_side = (self.WIDTH - self.puck.position[0]) / self.puck.velocity[0] if self.puck.velocity[0] > 0 else 0
            predicted_y_at_ai_side = self.puck.position[1] + self.puck.velocity[1] * time_to_ai_side
            predicted_y_norm = np.clip(predicted_y_at_ai_side / self.HEIGHT, 0, 1)
        else:
            predicted_y_norm = puck_y_norm
        
        # Construir vector de observación (21 dimensiones)
        observation = np.array([
            # Posiciones (6)
            ai_x_norm, ai_y_norm,
            puck_x_norm, puck_y_norm,
            human_x_norm, human_y_norm,
            
            # Velocidades (6)
            puck_vx_norm, puck_vy_norm,
            ai_vx_norm, ai_vy_norm,
            human_vx_norm, human_vy_norm,
            
            # Distancias (2)
            puck_to_ai_dist, puck_to_human_dist,
            
            # Información contextual (7)
            puck_in_ai_half,
            puck_moving_to_ai_goal,
            puck_moving_to_human_goal,
            time_factor,
            score_diff,
            predicted_y_norm,
            self.difficulty_level / 5.0  # Nivel de dificultad normalizado
        ], dtype=np.float32)
        
        return observation
    
    def _check_ai_collision(self):
        """Verificar colisión entre AI mallet y puck"""
        dx = self.puck.position[0] - self.ai_mallet_position[0]
        dy = self.puck.position[1] - self.ai_mallet_position[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance <= self.puck.radius + self.ai_mallet_radius:
            # Normalizar dirección
            if distance > 0:
                dx /= distance
                dy /= distance
            else:
                dx, dy = 1, 0
            
            # Reposicionar puck
            self.puck.position[0] = self.ai_mallet_position[0] + (self.ai_mallet_radius + self.puck.radius + 1) * dx
            self.puck.position[1] = self.ai_mallet_position[1] + (self.ai_mallet_radius + self.puck.radius + 1) * dy
            
            # Asegurar límites
            self.puck.position[0] = max(self.puck.radius, min(self.puck.position[0], self.WIDTH - self.puck.radius))
            self.puck.position[1] = max(self.puck.radius, min(self.puck.position[1], self.HEIGHT - self.puck.radius))
            
            # Transferir velocidad
            mallet_speed = np.sqrt(self.ai_mallet_velocity[0]**2 + self.ai_mallet_velocity[1]**2)
            base_speed = 8
            total_speed = base_speed + mallet_speed * 0.7
            
            self.puck.velocity[0] = dx * total_speed
            self.puck.velocity[1] = dy * total_speed
            
            # Actualizar rect del puck
            self.puck.rect.center = self.puck.position
            return True
        
        return False
    
    def render(self):
        """Renderizar el juego"""
        if self.render_mode is None:
            return
        
        # Inicializar display si es necesario
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                pygame.display.set_caption("Improved Air Hockey Training")
            else:
                self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Manejar eventos
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return False
        
        # Dibujar elementos del juego
        self.table.draw(self.screen)
        
        # Dibujar sprites
        all_sprites = pygame.sprite.Group()
        all_sprites.add(self.human_mallet, self.ai_mallet, self.puck)
        all_sprites.draw(self.screen)
        
        # Dibujar información
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"{self.player_score} - {self.ai_score}", True, self.WHITE)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 20))
        
        # Mostrar nivel de dificultad
        diff_text = font.render(f"Difficulty: {self.difficulty_level}", True, self.WHITE)
        self.screen.blit(diff_text, (10, 10))
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)
        
        return True
    
    def close(self):
        """Limpiar recursos"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

def create_improved_env():
    """Crear entorno mejorado"""
    env = ImprovedAirHockeyEnv()
    env = Monitor(env)
    return env

def train_improved_agent(total_timesteps=2000000, model_name="improved_air_hockey"):
    """Entrenar agente con sistema mejorado"""
    
    print("Creating improved training environment...")
    env = create_improved_env()
    eval_env = create_improved_env()
    
    # Crear directorios
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "improved_models")
    logs_dir = os.path.join(current_dir, "improved_logs")
    best_model_dir = os.path.join(models_dir, "best_model")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Callbacks mejorados
    checkpoint_callback = CheckpointCallback(
        save_freq=200000,
        save_path=models_dir,
        name_prefix=model_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=logs_dir,
        eval_freq=50000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    curriculum_callback = CurriculumLearningCallback(eval_env, eval_freq=100000, verbose=1)
    reward_callback = AdvancedRewardCallback(verbose=1)
    
    # Modelo PPO optimizado
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=2e-4,  # Learning rate más bajo para estabilidad
        n_steps=4096,        # Más pasos por actualización
        batch_size=128,      # Batch size más grande
        gamma=0.995,         # Factor de descuento más alto para planificación a largo plazo
        gae_lambda=0.98,     # GAE más alto para mejor estimación de ventaja
        clip_range=0.15,     # Clipping más conservador
        ent_coef=0.005,      # Menos entropía para comportamiento más determinista
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],  # Red más profunda
            activation_fn=torch.nn.ReLU
        ),
        verbose=1
    )
    
    print(f"Training improved agent for {total_timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, curriculum_callback, reward_callback],
        progress_bar=True
    )
    
    # Guardar modelo final
    final_model_path = f"./improved_models/{model_name}_final"
    model.save(final_model_path)
    
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Final model saved as {final_model_path}")
    
    # Evaluación final
    print("Evaluating final agent...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50)
    print(f"Final performance: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model

if __name__ == "__main__":
    print("=== Sistema de Entrenamiento Mejorado para Air Hockey ===")
    print("\nMejoras implementadas:")
    print("- Sistema de recompensas avanzado y balanceado")
    print("- Oponente inteligente con 6 niveles de dificultad")
    print("- Curriculum learning automático")
    print("- Observaciones expandidas (21 dimensiones)")
    print("- Hiperparámetros optimizados")
    print("- Métricas de rendimiento detalladas")
    
    choice = input("\n¿Entrenar nuevo modelo mejorado? (y/n): ").lower().startswith('y')
    
    if choice:
        timesteps = input("Timesteps de entrenamiento (default: 2000000): ").strip()
        timesteps = int(timesteps) if timesteps else 2000000
        
        model = train_improved_agent(total_timesteps=timesteps)
        print("\n¡Entrenamiento completado!")
        print("El modelo mejorado debería mostrar un rendimiento significativamente mejor.")
    else:
        print("Para usar este sistema, ejecuta el entrenamiento cuando estés listo.") 