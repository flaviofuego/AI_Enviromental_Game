import numpy as np
import random
import pygame
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from constants import NEON_GREEN, WIDTH, HEIGHT  # Añade esta línea

class DQNAgent:
    """Agente de aprendizaje por refuerzo Deep Q-Network"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # factor de descuento
        self.epsilon = 1.0   # tasa de exploración inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        """Construye la red neuronal para aproximar la función Q(s,a)"""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Almacena experiencias en la memoria"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Decide una acción basada en el estado actual"""
        if np.random.rand() <= self.epsilon:
            # Exploración: acción aleatoria
            return random.randrange(self.action_size)
        
        # Explotación: predecir acción según el modelo
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Entrena la red con experiencias pasadas"""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Verificar que state y next_state no sean None
            if state is None or next_state is None:
                continue
                
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, name):
        """Guarda el modelo"""
        self.model.save(name)
        
    def load(self, name):
        """Carga un modelo guardado"""
        self.model = load_model(name)

class RLAIMallet(pygame.sprite.Sprite):
    """Mallet controlado por el agente de RL"""
    
    def __init__(self, agent):
        super().__init__()
        self.radius = 30
        self.image = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, NEON_GREEN, (self.radius, self.radius), self.radius)
        pygame.draw.circle(self.image, (255, 255, 255, 150), (self.radius, self.radius), self.radius // 2)
        self.rect = self.image.get_rect(center=(WIDTH * 3 // 4, HEIGHT // 2))
        self.mask = pygame.mask.from_surface(self.image)
        self.velocity = [0, 0]
        self.position = [WIDTH * 3 // 4, HEIGHT // 2]
        self.prev_position = self.position.copy()
        self.agent = agent
        self.state = np.zeros((1, agent.state_size))
        
    def get_state(self, puck):
        """Obtiene el vector de estado para el agente RL"""
        return np.reshape([
            self.position[0] / WIDTH,
            self.position[1] / HEIGHT,
            puck.position[0] / WIDTH,
            puck.position[1] / HEIGHT,
            puck.velocity[0] / puck.max_speed,
            puck.velocity[1] / puck.max_speed
        ], [1, 6])
    
    def update(self, puck_pos, puck=None, training=False):
        """Actualiza la posición del mallet basado en las decisiones del agente RL"""
        self.prev_position = self.position.copy()
        
        if puck and puck_pos[0] > WIDTH // 2:  # Solo si la bola está en su mitad
            # Obtener estado actual
            current_state = self.get_state(puck)
            
            # Elegir acción (0: arriba, 1: abajo, 2: izquierda, 3: derecha, 4: no moverse)
            action = self.agent.act(current_state)
            
            # Aplicar acción
            move_amount = 5
            if action == 0:  # Arriba
                new_y = max(self.position[1] - move_amount, self.radius)
                self.position[1] = new_y
            elif action == 1:  # Abajo
                new_y = min(self.position[1] + move_amount, HEIGHT - self.radius)
                self.position[1] = new_y
            elif action == 2:  # Izquierda
                new_x = max(self.position[0] - move_amount, WIDTH // 2 + self.radius)
                self.position[0] = new_x
            elif action == 3:  # Derecha
                new_x = min(self.position[0] + move_amount, WIDTH - self.radius)
                self.position[0] = new_x
            # Si action == 4, no hacemos nada
            
            # Actualizar rect
            self.rect.center = self.position
            
            # Calcular velocidad
            self.velocity = [
                self.position[0] - self.prev_position[0],
                self.position[1] - self.prev_position[1]
            ]
            
            if training:
                # Guardar estado para entrenamiento
                self.state = current_state
                return action
        
        if not training and (puck is None or puck_pos[0] <= WIDTH // 2):
            # Comportamiento predeterminado cuando la bola no está en su mitad
            self.position[1] += (HEIGHT // 2 - self.position[1]) * 0.05
            self.rect.center = self.position
            
            # Actualizar velocidad
            self.velocity = [
                self.position[0] - self.prev_position[0],
                self.position[1] - self.prev_position[1]
            ]