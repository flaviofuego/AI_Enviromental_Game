# improved_training_system_fixed.py
import gymnasium as gym
import numpy as np
import time
import os
import pygame
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import deque

torch.set_num_threads(6)
torch.cuda.is_available = lambda: False

class BehaviorAnalysisCallback(BaseCallback):
    """Callback para analizar el comportamiento del agente durante el entrenamiento"""
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.action_history = deque(maxlen=5000)
        self.position_history = deque(maxlen=1000)
        self.episode_rewards = deque(maxlen=100)
        
    def _on_step(self):
        # Registrar acción
        if hasattr(self.locals, 'actions') and self.locals['actions'] is not None:
            action = self.locals['actions'][0]
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.ndim == 0 else int(action[0])
            else:
                action = int(action)
            self.action_history.append(action)
        
        # Analizar comportamiento cada 25,000 pasos
        if self.n_calls % 25000 == 0 and len(self.action_history) > 1000:
            self._analyze_behavior()
        
        return True
    
    def _analyze_behavior(self):
        actions = list(self.action_history)
        total = len(actions)
        
        up_count = sum(1 for a in actions if a == 0)
        down_count = sum(1 for a in actions if a == 1)
        left_count = sum(1 for a in actions if a == 2)
        right_count = sum(1 for a in actions if a == 3)
        stay_count = sum(1 for a in actions if a == 4)
        
        vertical_pct = ((up_count + down_count) / total) * 100
        horizontal_pct = ((left_count + right_count) / total) * 100
        stay_pct = (stay_count / total) * 100
        
        if self.verbose > 0:
            print(f"\n{'='*50}")
            print(f"BEHAVIOR ANALYSIS - Step {self.n_calls}")
            print(f"{'='*50}")
            print(f"Up: {up_count:>4} ({(up_count/total)*100:>5.1f}%)")
            print(f"Down: {down_count:>4} ({(down_count/total)*100:>5.1f}%)")
            print(f"Left: {left_count:>4} ({(left_count/total)*100:>5.1f}%)")
            print(f"Right: {right_count:>4} ({(right_count/total)*100:>5.1f}%)")
            print(f"Stay: {stay_count:>4} ({stay_pct:>5.1f}%)")
            print(f"Vertical Movement: {vertical_pct:.1f}%")
            print(f"Horizontal Movement: {horizontal_pct:.1f}%")
            
            if vertical_pct < 10:
                print("❌ CRÍTICO: Muy poco movimiento vertical!")
            elif vertical_pct < 20:
                print("⚠️  ADVERTENCIA: Poco movimiento vertical")
            elif vertical_pct > 15:
                print("✅ BIEN: Buen balance de movimiento vertical")

class FixedAirHockeyEnv(gym.Env):
    """Entorno de Air Hockey con sistema de recompensas completamente corregido"""
    
    def __init__(self, render_mode=None, play_mode=False):
        super().__init__()
        
        # Importar constantes y clases necesarias
        from constants import WIDTH, HEIGHT, WHITE
        from sprites import Puck, HumanMallet, AIMallet
        from table import Table
        
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.WHITE = WHITE
        
        # Inicializar pygame si es necesario
        if not pygame.get_init():
            pygame.init()
        
        # Espacios de acción y observación
        self.action_space = gym.spaces.Discrete(5)  # [Up, Down, Left, Right, Stay]
        
        # Observación ampliada: 21 dimensiones
        low = np.array([0] * 21, dtype=np.float32)
        high = np.array([1] * 21, dtype=np.float32)
        # Velocidades pueden ser negativas
        low[6:12] = -1.0
        high[6:12] = 1.0
        # Información contextual puede ser negativa
        low[14:17] = -1.0
        high[14:17] = 1.0
        low[18] = -1.0  # score_diff
        high[18] = 1.0
        
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        
        # Configuración de renderizado
        self.render_mode = render_mode
        self.play_mode = play_mode
        self.screen = None
        self.clock = None
        
        # Crear objetos del juego
        self.table = Table()
        self.puck = Puck()
        self.human_mallet = HumanMallet()
        self.ai_mallet = AIMallet()
        
        # Variables de estado del juego
        self.player_score = 0
        self.ai_score = 0
        self.steps = 0
        self.max_steps = 3000
        
        # Variables de posición y velocidad
        self.ai_mallet_position = [WIDTH * 3 // 4, HEIGHT // 2]
        self.ai_mallet_velocity = [0, 0]
        self.ai_mallet_radius = 30
        
        # Métricas para recompensas mejoradas
        self.last_distance_to_puck = 0
        self.consecutive_hits = 0
        self.time_since_last_hit = 0
        self.last_action = 4
        self.movement_history = deque(maxlen=20)
        self.position_history = deque(maxlen=10)
        
        # Configuración del oponente
        self.difficulty_level = 0
        self.opponent_skill = 0.3
        
        # Estadísticas del episodio
        self.episode_stats = {
            'hits': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'vertical_movements': 0,
            'avg_distance_to_puck': 0
        }
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reiniciar el entorno"""
        super().reset(seed=seed)
        
        # Reiniciar puntuaciones
        self.player_score = 0
        self.ai_score = 0
        self.steps = 0
        
        # Reiniciar posiciones
        self.puck.reset()
        self.human_mallet.position = [self.WIDTH // 4, self.HEIGHT // 2]
        self.human_mallet.rect.center = self.human_mallet.position
        self.ai_mallet_position = [self.WIDTH * 3 // 4, self.HEIGHT // 2]
        self.ai_mallet.rect.center = self.ai_mallet_position
        
        # Reiniciar velocidades
        self.human_mallet.velocity = [0, 0]
        self.ai_mallet_velocity = [0, 0]
        
        # Reiniciar métricas
        self.last_distance_to_puck = self._calculate_distance_to_puck()
        self.consecutive_hits = 0
        self.time_since_last_hit = 0
        self.last_action = 4
        self.movement_history.clear()
        self.position_history.clear()
        
        # Reiniciar estadísticas
        self.episode_stats = {
            'hits': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'vertical_movements': 0,
            'avg_distance_to_puck': 0
        }
        
        observation = self._get_observation()
        info = {"player_score": 0, "ai_score": 0}
        
        return observation, info
    
    def step(self, action):
        """Ejecutar un paso en el entorno"""
        self.steps += 1
        self.time_since_last_hit += 1
        
        # Guardar estado previo
        prev_position = self.ai_mallet_position.copy()
        prev_distance = self._calculate_distance_to_puck()
        
        # Aplicar acción del AI
        self._apply_action(action)
        
        # Rastrear movimiento
        self.movement_history.append(action)
        self.position_history.append(self.ai_mallet_position.copy())
        
        # Contar movimientos verticales
        if action in [0, 1]:
            self.episode_stats['vertical_movements'] += 1
        
        # Actualizar oponente humano (solo en entrenamiento)
        if not self.play_mode:
            self._update_simple_opponent()
        
        # Actualizar física del puck
        self.puck.update()
        
        # Verificar colisiones
        ai_hit = self._check_ai_collision()
        human_hit = self.puck.check_mallet_collision(self.human_mallet)
        
        if ai_hit:
            self.consecutive_hits += 1
            self.time_since_last_hit = 0
            self.episode_stats['hits'] += 1
        else:
            self.consecutive_hits = 0
        
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
        
        # Calcular recompensa corregida
        reward = self._calculate_fixed_reward(prev_distance, ai_hit, goal, action, prev_position)
        
        # Verificar fin del episodio
        done = (goal_scored or self.steps >= self.max_steps or 
                self.player_score >= 7 or self.ai_score >= 7)
        
        truncated = self.steps >= self.max_steps and not goal_scored
        
        # Obtener observación
        observation = self._get_observation()
        
        # Información adicional
        info = {
            "player_score": self.player_score,
            "ai_score": self.ai_score,
            "steps": self.steps,
            "hit_puck": ai_hit,
            "episode_stats": self.episode_stats.copy() if done else {}
        }
        
        return observation, reward, done, truncated, info
    
    def _calculate_fixed_reward(self, prev_distance, ai_hit, goal, action, prev_position):
        """Sistema de recompensas completamente corregido"""
        reward = 0.0
        
        # 1. RECOMPENSAS POR GOLES (Más importantes)
        if goal == "ai":
            reward += 100.0  # Gran recompensa por anotar
        elif goal == "player":
            reward -= 80.0   # Fuerte penalización por recibir gol
        
        # 2. RECOMPENSAS POR GOLPEAR EL PUCK
        if ai_hit:
            base_reward = 15.0
            
            # Bonus por dirección del golpe
            puck_direction = np.array(self.puck.velocity)
            if np.linalg.norm(puck_direction) > 0:
                # Recompensar golpear hacia la portería rival
                to_goal = np.array([-1, 0])  # Hacia la izquierda
                direction_norm = puck_direction / np.linalg.norm(puck_direction)
                direction_bonus = max(0, np.dot(direction_norm, to_goal)) * 10.0
                reward += base_reward + direction_bonus
            else:
                reward += base_reward
        
        # 3. RECOMPENSAS POR MOVIMIENTO (CRÍTICAS PARA CORREGIR COMPORTAMIENTO)
        current_distance = self._calculate_distance_to_puck()
        
        # FUERTES INCENTIVOS PARA MOVIMIENTO VERTICAL
        if action in [0, 1]:  # Up o Down
            reward += 0.5  # GRAN bonus por movimiento vertical
            
            # Bonus extra si se alinea con el puck verticalmente
            y_alignment = abs(self.puck.position[1] - self.ai_mallet_position[1])
            if y_alignment < 60:
                reward += 0.3
        
        # Incentivo moderado para movimiento horizontal estratégico
        elif action in [2, 3]:  # Left o Right
            reward += 0.2
        
        # Penalización por quedarse quieto excesivamente
        elif action == 4:  # Stay
            stay_penalty = 0.1
            
            # Penalización extra si el puck está cerca
            if current_distance < 80:
                stay_penalty += 0.2
            
            reward -= stay_penalty
        
        # 4. RECOMPENSAS POR POSICIONAMIENTO DINÁMICO
        
        # Recompensar acercarse al puck cuando está en el lado del AI
        if self.puck.position[0] > self.WIDTH // 2:
            if current_distance < prev_distance:
                reward += 0.3  # Recompensa por acercarse
            elif current_distance > prev_distance + 5:
                reward -= 0.1  # Penalización por alejarse sin razón
        
        # Posicionamiento defensivo DINÁMICO (no estático)
        elif self.puck.position[0] < self.WIDTH // 2:
            # Posición defensiva ideal más AGRESIVA
            ideal_defensive_x = self.WIDTH * 0.65  # Más adelante, no tan atrás
            ideal_defensive_y = self.puck.position[1]
            
            current_x = self.ai_mallet_position[0]
            current_y = self.ai_mallet_position[1]
            
            # Recompensar estar en la posición X correcta
            x_distance = abs(current_x - ideal_defensive_x)
            if x_distance < 40:
                reward += 0.2
            
            # Recompensar seguir al puck verticalmente
            y_distance = abs(current_y - ideal_defensive_y)
            if y_distance < 50:
                reward += 0.3
            elif y_distance < 100:
                reward += 0.1
        
        # 5. RECOMPENSAS POR CONTROL DEL JUEGO
        
        # Bonus por tener el puck en el lado del oponente
        if self.puck.position[0] < self.WIDTH // 2:
            reward += 0.15
        
        # Bonus por velocidad del puck hacia la portería rival
        if self.puck.velocity[0] < -2.0:
            reward += 0.25
        
        # 6. PENALIZACIONES POR COMPORTAMIENTOS PROBLEMÁTICOS
        
        # Fuerte penalización por quedarse en el fondo de la cancha
        if self.ai_mallet_position[1] > self.HEIGHT * 0.8:  # Muy abajo
            reward -= 0.4
        elif self.ai_mallet_position[1] < self.HEIGHT * 0.2:  # Muy arriba
            reward -= 0.4
        
        # Penalización por no tocar el puck por mucho tiempo
        if self.time_since_last_hit > 150:
            timeout_penalty = min((self.time_since_last_hit - 150) * 0.02, 3.0)
            reward -= timeout_penalty
        
        # 7. BONUS POR PATRONES DE MOVIMIENTO SALUDABLES
        
        if len(self.movement_history) >= 10:
            recent_actions = list(self.movement_history)[-10:]
            
            # Contar tipos de movimiento
            vertical_count = sum(1 for a in recent_actions if a in [0, 1])
            horizontal_count = sum(1 for a in recent_actions if a in [2, 3])
            stay_count = sum(1 for a in recent_actions if a == 4)
            
            # Bonus por usar movimiento vertical
            if vertical_count >= 2:
                reward += 0.2
            
            # Penalización por solo moverse horizontalmente
            if vertical_count == 0 and horizontal_count > 6:
                reward -= 0.3
            
            # Penalización por quedarse quieto demasiado
            if stay_count > 7:
                reward -= 0.4
        
        # 8. EXPLORACIÓN DE POSICIONES
        if len(self.position_history) >= 5:
            positions = list(self.position_history)[-5:]
            y_positions = [pos[1] for pos in positions]
            y_variance = np.var(y_positions)
            
            # Recompensar exploración vertical
            if y_variance > 500:  # Buena variación en Y
                reward += 0.25
            elif y_variance > 200:
                reward += 0.1
        
        return reward
    
    def _apply_action(self, action):
        """Aplicar acción del AI con movimiento más rápido"""
        move_amount = 10  # Movimiento más ágil
        
        prev_position = self.ai_mallet_position.copy()
        
        if action == 0:  # Up
            self.ai_mallet_position[1] = max(
                self.ai_mallet_position[1] - move_amount, 
                self.ai_mallet_radius
            )
        elif action == 1:  # Down
            self.ai_mallet_position[1] = min(
                self.ai_mallet_position[1] + move_amount, 
                self.HEIGHT - self.ai_mallet_radius
            )
        elif action == 2:  # Left
            self.ai_mallet_position[0] = max(
                self.ai_mallet_position[0] - move_amount, 
                self.WIDTH // 2 + self.ai_mallet_radius
            )
        elif action == 3:  # Right
            self.ai_mallet_position[0] = min(
                self.ai_mallet_position[0] + move_amount, 
                self.WIDTH - self.ai_mallet_radius
            )
        # action == 4 es "stay"
        
        # Actualizar sprite
        self.ai_mallet.rect.center = self.ai_mallet_position
        
        # Calcular velocidad
        self.ai_mallet_velocity = [
            self.ai_mallet_position[0] - prev_position[0],
            self.ai_mallet_position[1] - prev_position[1]
        ]
    
    def _update_simple_opponent(self):
        """Oponente humano simple pero efectivo"""
        # Movimiento simple hacia el puck con algo de inercia
        target_x = max(self.puck.position[0] - 30, self.human_mallet.radius)
        target_x = min(target_x, self.WIDTH // 2 - self.human_mallet.radius)
        target_y = self.puck.position[1]
        
        # Limitar velocidad del oponente
        max_speed = 8 * self.opponent_skill
        
        dx = target_x - self.human_mallet.position[0]
        dy = target_y - self.human_mallet.position[1]
        
        distance = np.sqrt(dx**2 + dy**2)
        if distance > max_speed:
            dx = (dx / distance) * max_speed
            dy = (dy / distance) * max_speed
        
        # Actualizar posición
        new_x = np.clip(
            self.human_mallet.position[0] + dx,
            self.human_mallet.radius,
            self.WIDTH // 2 - self.human_mallet.radius
        )
        new_y = np.clip(
            self.human_mallet.position[1] + dy,
            self.human_mallet.radius,
            self.HEIGHT - self.human_mallet.radius
        )
        
        self.human_mallet.position = [new_x, new_y]
        self.human_mallet.rect.center = self.human_mallet.position
        self.human_mallet.velocity = [dx, dy]
    
    def _get_observation(self):
        """Obtener observación de 21 dimensiones"""
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
        score_diff = (self.ai_score - self.player_score) / 7.0
        
        # Predicción de trayectoria
        if abs(self.puck.velocity[0]) > 0.1:
            time_to_ai_side = (self.WIDTH - self.puck.position[0]) / self.puck.velocity[0] if self.puck.velocity[0] > 0 else 0
            predicted_y = self.puck.position[1] + self.puck.velocity[1] * time_to_ai_side
            predicted_y_norm = np.clip(predicted_y / self.HEIGHT, 0, 1)
        else:
            predicted_y_norm = puck_y_norm
        
        # Vector de observación completo (21 dimensiones)
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
            0.5  # Nivel de dificultad normalizado
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_distance_to_puck(self):
        """Calcular distancia del AI mallet al puck"""
        return np.sqrt(
            (self.puck.position[0] - self.ai_mallet_position[0])**2 + 
            (self.puck.position[1] - self.ai_mallet_position[1])**2
        )
    
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
            
            # Transferir velocidad con la velocidad del mallet
            mallet_speed = np.sqrt(self.ai_mallet_velocity[0]**2 + self.ai_mallet_velocity[1]**2)
            base_speed = 10
            total_speed = base_speed + mallet_speed * 0.8
            
            self.puck.velocity[0] = dx * total_speed
            self.puck.velocity[1] = dy * total_speed
            
            # Actualizar rect del puck
            self.puck.rect.center = self.puck.position
            return True
        
        return False

def create_fixed_env():
    """Crear entorno corregido"""
    env = FixedAirHockeyEnv()
    env = Monitor(env)
    return env

def train_fixed_agent(total_timesteps=1500000, model_name="fixed_air_hockey"):
    """Entrenar agente con sistema completamente corregido"""
    
    print("🔧 CREATING FIXED TRAINING ENVIRONMENT...")
    print("✅ Correcciones implementadas:")
    print("   - Incentivos fuertes para movimiento vertical")
    print("   - Penalizaciones por quedarse en el fondo")
    print("   - Posicionamiento defensivo más agresivo")
    print("   - Recompensas balanceadas para atacar y defender")
    print("   - Eliminación de penalizaciones contraproducentes")
    
    env = create_fixed_env()
    eval_env = create_fixed_env()
    
    # Crear directorios
    models_dir = "improved_models"
    logs_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Callbacks
    behavior_callback = BehaviorAnalysisCallback(verbose=1)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=150000,
        save_path=models_dir,
        name_prefix=model_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{models_dir}/best_{model_name}",
        log_path=logs_dir,
        eval_freq=75000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    # Modelo PPO optimizado para exploración y balance
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=3e-4,  # Learning rate un poco más alto para exploración
        n_steps=2048,
        batch_size=128,
        gamma=0.99,          # Gamma un poco más bajo para enfocarse en recompensas inmediatas
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,       # Entropía más alta para más exploración
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],
            activation_fn=torch.nn.ReLU
        ),
        verbose=1
    )
    
    print(f"\n🚀 TRAINING FIXED AGENT FOR {total_timesteps} TIMESTEPS...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[behavior_callback, checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Calcular tiempo
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n✅ TRAINING COMPLETED in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Guardar modelo final
    final_model_path = f"{models_dir}/{model_name}_final"
    model.save(final_model_path)
    print(f"💾 FIXED MODEL SAVED: {final_model_path}")
    
    # Evaluación final
    print("\n📊 EVALUATING FINAL AGENT...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"Final performance: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model

def test_fixed_model_behavior(model_path, num_tests=2000):
    """Probar el comportamiento del modelo corregido"""
    print(f"\n🧪 TESTING FIXED MODEL: {model_path}")
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    env = FixedAirHockeyEnv()
    
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    action_names = ["Up", "Down", "Left", "Right", "Stay"]
    
    position_samples = []
    
    obs, _ = env.reset()
    
    for i in range(num_tests):
        action, _ = model.predict(obs, deterministic=True)
        
        # Convertir acción a int
        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.ndim == 0 else int(action[0])
        else:
            action = int(action)
        
        action_counts[action] += 1
        
        # Muestrear posiciones
        if i % 50 == 0:
            position_samples.append(env.ai_mallet_position[1])
        
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            obs, _ = env.reset()
    
    # Análisis completo
    print(f"\n{'='*60}")
    print("ANÁLISIS DEL MODELO CORREGIDO")
    print(f"{'='*60}")
    
    total = sum(action_counts.values())
    print(f"\nDistribución de Acciones ({num_tests} pasos):")
    for action_id in range(5):
        count = action_counts[action_id]
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {action_names[action_id]:>6}: {count:>4} ({pct:>5.1f}%)")
    
    vertical_actions = action_counts[0] + action_counts[1]
    horizontal_actions = action_counts[2] + action_counts[3]
    
    print(f"\nResumen de Movimiento:")
    print(f"  🔺 Vertical: {(vertical_actions/total)*100:.1f}%")
    print(f"  ↔️ Horizontal: {(horizontal_actions/total)*100:.1f}%")
    print(f"  ⏸️  Stay: {(action_counts[4]/total)*100:.1f}%")
    
    # Análisis de posiciones
    if position_samples:
        avg_y = np.mean(position_samples)
        height_pct = (avg_y / env.HEIGHT) * 100
        y_variance = np.var(position_samples)
        
        print(f"\nAnálisis Posicional:")
        print(f"  Posición Y promedio: {avg_y:.1f} ({height_pct:.1f}% del campo)")
        print(f"  Varianza vertical: {y_variance:.1f}")
        
        if height_pct > 70:
            print("  ⚠️  ADVERTENCIA: Tiende a quedarse en la parte inferior")
        elif height_pct < 30:
            print("  ⚠️  ADVERTENCIA: Tiende a quedarse en la parte superior")
        else:
            print("  ✅ BIEN: Posicionamiento centrado")
    
    # Evaluación final
    print(f"\nEvaluación General:")
    if vertical_actions == 0:
        print("  ❌ CRÍTICO: Sin movimiento vertical")
    elif vertical_actions < total * 0.1:
        print("  ⚠️  PROBLEMA: Muy poco movimiento vertical")
    elif vertical_actions < total * 0.2:
        print("  📈 MEJORANDO: Movimiento vertical moderado")
    elif vertical_actions < total * 0.4:
        print("  ✅ BIEN: Buen balance de movimiento vertical")
    else:
        print("  🏆 EXCELENTE: Movimiento muy equilibrado")

if __name__ == "__main__":
    print("🔧 SISTEMA DE ENTRENAMIENTO CORREGIDO PARA AIR HOCKEY")
    print("=" * 60)
    print("\nEste sistema soluciona los problemas identificados:")
    print("❌ AI se queda pegado en el fondo → ✅ Incentivos para movimiento dinámico")
    print("❌ No usa movimiento vertical → ✅ Fuertes recompensas para Up/Down")
    print("❌ No ataca efectivamente → ✅ Recompensas balanceadas ofensiva/defensiva")
    print("❌ Defiende pasivamente → ✅ Posicionamiento defensivo más agresivo")
    
    print("\nOpciones:")
    print("1. Entrenar modelo corregido (1.5M timesteps)")
    print("2. Entrenar modelo corregido rápido (750K timesteps)")
    print("3. Probar modelo existente")
    
    choice = input("\nSelecciona opción (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Entrenando modelo corregido completo...")
        model = train_fixed_agent(1500000, "fixed_air_hockey")
        
        print("\n🧪 Probando el nuevo modelo...")
        test_fixed_model_behavior(f"improved_models/fixed_air_hockey_final.zip")
        
    elif choice == "2":
        print("\n⚡ Entrenando modelo corregido rápido...")
        model = train_fixed_agent(750000, "quick_fixed_model")
        
        print("\n🧪 Probando el nuevo modelo...")
        test_fixed_model_behavior(f"improved_models/quick_fixed_model_final.zip")
        
    elif choice == "3":
        # Buscar modelos disponibles
        models = [
            "improved_models/fixed_air_hockey_final.zip",
            "improved_models/quick_fixed_model_final.zip",
            "improved_models/enhanced_vertical_model_final.zip",
            "improved_models/quick_enhanced_model_final.zip",
            "improved_models/improved_air_hockey_final.zip"
        ]
        
        available = [m for m in models if os.path.exists(m)]
        
        if not available:
            print("❌ No se encontraron modelos para probar")
        else:
            print("\nModelos disponibles:")
            for i, model in enumerate(available):
                print(f"{i+1}. {model}")
            
            try:
                idx = int(input(f"Selecciona modelo (1-{len(available)}): ")) - 1
                if 0 <= idx < len(available):
                    test_fixed_model_behavior(available[idx])
                else:
                    print("❌ Selección inválida")
            except ValueError:
                print("❌ Entrada inválida")
    else:
        print("❌ Opción inválida") 