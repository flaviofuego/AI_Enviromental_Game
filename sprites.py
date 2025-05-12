import pygame
import math
import random
import numpy as np
from constants import *
from utils import calculate_vector, normalize_vector, vector_length, line_circle_intersection

class Mallet(pygame.sprite.Sprite):
    """Clase base para los mallets (mazos)"""
    
    def __init__(self, x, y, color):
        super().__init__()
        self.radius = 30
        self.image = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, color, (self.radius, self.radius), self.radius)
        # Añadir brillo al centro
        pygame.draw.circle(self.image, (255, 255, 255, 150), (self.radius, self.radius), self.radius // 2)
        self.rect = self.image.get_rect(center=(x, y))
        self.mask = pygame.mask.from_surface(self.image)
        self.velocity = [0, 0]
        self.position = [x, y]
        self.prev_position = self.position.copy()
        
    def update(self, mouse_pos=None):
        self.prev_position = self.position.copy()
        if mouse_pos:
            # Limitar el movimiento a la mitad de la pantalla para el jugador humano
            target_x = min(max(mouse_pos[0], self.radius), WIDTH - self.radius)
            target_y = min(max(mouse_pos[1], self.radius), HEIGHT - self.radius)
            
            # Restringir jugador a su mitad del campo
            if isinstance(self, HumanMallet):
                target_x = min(max(mouse_pos[0], self.radius), WIDTH // 2 - self.radius)
            else:
                target_x = min(max(mouse_pos[0], WIDTH // 2 + self.radius), WIDTH - self.radius)
            
            self.position = [target_x, target_y]
            self.rect.center = self.position
            
            # Calcular velocidad basada en el cambio de posición
            self.velocity = [
                self.position[0] - self.prev_position[0],
                self.position[1] - self.prev_position[1]
            ]

class HumanMallet(Mallet):
    """Mallet controlado por el jugador humano"""
    
    def __init__(self):
        super().__init__(WIDTH // 4, HEIGHT // 2, NEON_RED)

class AIMallet(Mallet):
    """Mallet controlado por la IA simple"""
    
    def __init__(self):
        super().__init__(WIDTH * 3 // 4, HEIGHT // 2, NEON_GREEN)
        
    def update(self, puck_pos):
        self.prev_position = self.position.copy()
        
        # Movimiento simple de seguimiento, se reemplazará con RL
        if puck_pos:
            if puck_pos[0] > WIDTH // 2:  # Solo si la bola está en su mitad
                # Detectar si el puck está cerca de una esquina
                is_near_corner = (
                    (puck_pos[0] < self.radius*3 or puck_pos[0] > WIDTH - self.radius*3) and 
                    (puck_pos[1] < self.radius*3 or puck_pos[1] > HEIGHT - self.radius*3)
                )
                
                if is_near_corner:
                    # Estrategia de esquina: moverse hacia el centro en lugar de hacia el puck
                    # Calcular vector desde la esquina hacia el centro de la mesa
                    target_x = WIDTH * 3 // 4  # Punto central de la mitad derecha
                    target_y = HEIGHT // 2
                    
                    # Moverse hacia ese punto con suavidad
                    self.position[0] += (target_x - self.position[0]) * 0.15
                    self.position[1] += (target_y - self.position[1]) * 0.15
                else:
                    # Comportamiento normal con algo de inteligencia añadida
                    # Intentar interceptar el puck basado en su trayectoria, no solo su posición
                    # Calcular punto de interceptación estimado
                    intercept_x = min(max(puck_pos[0] + random.randint(-20, 20), WIDTH // 2 + self.radius), WIDTH - self.radius)
                    intercept_y = min(max(puck_pos[1] + random.randint(-20, 20), self.radius), HEIGHT - self.radius)
                    
                    # Mover hacia el punto de interceptación con suavidad
                    self.position[0] += (intercept_x - self.position[0]) * 0.1
                    self.position[1] += (intercept_y - self.position[1]) * 0.1
                
                self.rect.center = self.position
                
                # Calcular velocidad basada en el cambio de posición
                self.velocity = [
                    self.position[0] - self.prev_position[0],
                    self.position[1] - self.prev_position[1]
                ]
class Puck(pygame.sprite.Sprite):
    """Clase para el disco (puck)"""
    
    def __init__(self):
        super().__init__()
        self.radius = 15
        self.image = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, NEON_BLUE, (self.radius, self.radius), self.radius)
        # Añadir brillo
        pygame.draw.circle(self.image, (255, 255, 255, 150), (self.radius, self.radius), self.radius // 2)
        self.rect = self.image.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        self.mask = pygame.mask.from_surface(self.image)
        self.velocity = [random.choice([-2, 2]), random.choice([-2, 2])]
        self.position = [WIDTH // 2, HEIGHT // 2]
        self.prev_position = self.position.copy()
        self.max_speed = MAX_PUCK_SPEED
        self.friction = FRICTION
        
    def update(self):
        self.prev_position = self.position.copy()
        
        # Aplicar fricción
        self.velocity[0] *= self.friction
        self.velocity[1] *= self.friction
        
        # Limitar velocidad máxima
        speed = vector_length(self.velocity)
        if speed > self.max_speed:
            normalized = normalize_vector(self.velocity)
            self.velocity[0] = normalized[0] * self.max_speed
            self.velocity[1] = normalized[1] * self.max_speed
        
        # Guardar la posición anterior para cálculos
        old_x, old_y = self.position
        
        # Calcular nueva posición
        new_x = old_x + self.velocity[0]
        new_y = old_y + self.velocity[1]
        
        # Verificar colisiones con las paredes y manejar cada eje por separado
        # Primero colisión horizontal (izquierda/derecha)
        if new_x - self.radius < 0:
            new_x = self.radius
            self.velocity[0] = abs(self.velocity[0]) * COLLISION_ELASTICITY  # Asegurar rebote hacia adentro
        elif new_x + self.radius > WIDTH:
            new_x = WIDTH - self.radius
            self.velocity[0] = -abs(self.velocity[0]) * COLLISION_ELASTICITY  # Asegurar rebote hacia adentro
            
        # Luego colisión vertical (arriba/abajo)
        if new_y - self.radius < 0:
            new_y = self.radius
            self.velocity[1] = abs(self.velocity[1]) * COLLISION_ELASTICITY  # Asegurar rebote hacia adentro
        elif new_y + self.radius > HEIGHT:
            new_y = HEIGHT - self.radius
            self.velocity[1] = -abs(self.velocity[1]) * COLLISION_ELASTICITY  # Asegurar rebote hacia adentro
        
        # Verificación especial para las esquinas (añadir un rebote más fuerte)
        corner_radius = self.radius * 1.5  # Radio para detectar proximidad a esquinas
        
        # Esquina superior izquierda
        if new_x < corner_radius and new_y < corner_radius:
            # Aplicar una fuerza que aleje el puck de la esquina
            force_direction = normalize_vector([1, 1])
            force_magnitude = 0.5
            self.velocity[0] += force_direction[0] * force_magnitude
            self.velocity[1] += force_direction[1] * force_magnitude
            
        # Esquina superior derecha
        elif new_x > WIDTH - corner_radius and new_y < corner_radius:
            force_direction = normalize_vector([-1, 1])
            force_magnitude = 0.5
            self.velocity[0] += force_direction[0] * force_magnitude
            self.velocity[1] += force_direction[1] * force_magnitude
            
        # Esquina inferior izquierda
        elif new_x < corner_radius and new_y > HEIGHT - corner_radius:
            force_direction = normalize_vector([1, -1])
            force_magnitude = 0.5
            self.velocity[0] += force_direction[0] * force_magnitude
            self.velocity[1] += force_direction[1] * force_magnitude
            
        # Esquina inferior derecha
        elif new_x > WIDTH - corner_radius and new_y > HEIGHT - corner_radius:
            force_direction = normalize_vector([-1, -1])
            force_magnitude = 0.5
            self.velocity[0] += force_direction[0] * force_magnitude
            self.velocity[1] += force_direction[1] * force_magnitude
        
        # Garantía absoluta de que el puck nunca sale del campo de juego
        new_x = max(self.radius, min(new_x, WIDTH - self.radius))
        new_y = max(self.radius, min(new_y, HEIGHT - self.radius))
        
        # Asignar la nueva posición
        self.position = [new_x, new_y]
        self.rect.center = self.position
            
            
        
    def reset(self, scorer=None, zero_velocity=False):
        """
        Resetea el puck al centro con una velocidad determinada
        
        Args:
            scorer: Quién anotó el último gol ('player', 'ai', o None)
            zero_velocity: Si es True, establece la velocidad a cero
        """
        self.position = [WIDTH // 2, HEIGHT // 2]
        
        if zero_velocity:
            self.velocity = [0, 0]
        else:
            # Dirección aleatoria, pero ligeramente hacia el que no anotó
            if scorer == "player":
                self.velocity = [random.uniform(-3, -1), random.uniform(-2, 2)]
            elif scorer == "ai":
                self.velocity = [random.uniform(1, 3), random.uniform(-2, 2)]
            else:
                self.velocity = [random.choice([-2, 2]), random.choice([-2, 2])]
        
        self.rect.center = self.position
        
    def check_mallet_collision(self, mallet):
        """Verificación avanzada de colisión con un mallet, con prevención de atravesar paredes"""
        # Verificar si la trayectoria del puck intersecta con el mallet
        trajectory_intersects = line_circle_intersection(
            self.prev_position, 
            self.position, 
            mallet.position, 
            mallet.radius + self.radius
        )
        
        # Comprobación de colisión estándar
        standard_collision = pygame.sprite.collide_mask(self, mallet)
        
        if trajectory_intersects or standard_collision:
            # Calcular el vector desde el centro del mallet al centro del puck
            dx = self.position[0] - mallet.position[0]
            dy = self.position[1] - mallet.position[1]
            
            # Normalizar
            length = vector_length([dx, dy])
            if length > 0:
                dx /= length
                dy /= length
                
                # Verificar si estamos cerca de un borde para ajustar el vector de rebote
                near_left = self.position[0] - self.radius < 20
                near_right = self.position[0] + self.radius > WIDTH - 20
                near_top = self.position[1] - self.radius < 20
                near_bottom = self.position[1] + self.radius > HEIGHT - 20
                
                # Modificar el vector de rebote para evitar empujar contra los bordes
                if near_left:
                    dx = abs(dx)  # Forzar componente x positivo (hacia la derecha)
                elif near_right:
                    dx = -abs(dx)  # Forzar componente x negativo (hacia la izquierda)
                    
                if near_top:
                    dy = abs(dy)  # Forzar componente y positivo (hacia abajo)
                elif near_bottom:
                    dy = -abs(dy)  # Forzar componente y negativo (hacia arriba)
                
                # Reposicionar el puck fuera del mallet
                self.position[0] = mallet.position[0] + (mallet.radius + self.radius + 1) * dx
                self.position[1] = mallet.position[1] + (mallet.radius + self.radius + 1) * dy
                
                # Garantía absoluta de que el puck no salga del área jugable
                self.position[0] = max(self.radius, min(self.position[0], WIDTH - self.radius))
                self.position[1] = max(self.radius, min(self.position[1], HEIGHT - self.radius))
                
                # Transferir parte de la velocidad del mallet al puck
                mallet_speed_contribution = vector_length(mallet.velocity) * 0.5
                
                # Calcular la nueva velocidad del puck
                # Combina la reflexión con la velocidad del mallet
                self.velocity[0] = dx * (6 + mallet_speed_contribution)
                self.velocity[1] = dy * (6 + mallet_speed_contribution)
                
                self.rect.center = self.position
                return True
                
        return False