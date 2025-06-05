import pygame
import math
import random
import numpy as np
from constants import *
from utils import calculate_vector, normalize_vector, vector_length, line_circle_intersection

class Mallet(pygame.sprite.Sprite):
    """Clase base para los mallets (mazos)"""
    
    def __init__(self, x, y, color, custom_image=None):
        super().__init__()
        # Calcular radio escalado basado en las dimensiones actuales
        scale_factor = min(get_screen_dimensions()[0]/800, get_screen_dimensions()[1]/500)
        self.radius = int(MULLET_RADIUS * scale_factor)
        
        if custom_image is not None:
            # Usar imagen personalizada
            self.image = custom_image
            # Asegurar que la imagen tenga el tamaño correcto
            if self.image.get_size() != (self.radius * 2, self.radius * 2):
                self.image = pygame.transform.smoothscale(self.image, (self.radius * 2, self.radius * 2))
        else:
            # Crear sprite por defecto
            self.image = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(self.image, color, (self.radius, self.radius), self.radius)
            # Añadir brillo al centro
            pygame.draw.circle(self.image, (255, 255, 255, 150), (self.radius, self.radius), self.radius // 2)
        
        self.rect = self.image.get_rect(center=(x, y))
        # Crear máscara circular para colisiones precisas
        self.mask = self._create_circular_mask()
        self.velocity = [0, 0]
        self.position = [x, y]
        self.prev_position = self.position.copy()
    
    def _create_circular_mask(self):
        """Crea una máscara circular para colisiones precisas"""
        mask_surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(mask_surface, (255, 255, 255, 255), (self.radius, self.radius), self.radius)
        return pygame.mask.from_surface(mask_surface)
    
    def update(self, mouse_pos=None):
        current_width, current_height = get_screen_dimensions()
        self.prev_position = self.position.copy()
        
        if mouse_pos:
            # Limitar el movimiento a la pantalla actual
            target_x = min(max(mouse_pos[0], self.radius), current_width - self.radius)
            target_y = min(max(mouse_pos[1], self.radius), current_height - self.radius)
            
            # Restringir jugador a su mitad del campo
            if isinstance(self, HumanMallet):
                target_x = min(max(mouse_pos[0], self.radius), current_width // 2 - self.radius)
            else:
                target_x = min(max(mouse_pos[0], current_width // 2 + self.radius), current_width - self.radius)
            
            self.position = [target_x, target_y]
            self.rect.center = self.position
            
            # Calcular velocidad basada en el cambio de posición
            self.velocity = [
                self.position[0] - self.prev_position[0],
                self.position[1] - self.prev_position[1]
            ]

class HumanMallet(Mallet):
    def __init__(self, color=NEON_RED, custom_image=None):
        current_width, current_height = get_screen_dimensions()
        super().__init__(current_width // 4, current_height // 2, color, custom_image)


class AIMallet(Mallet):
    """Mallet controlado por la IA simple"""
    
    def __init__(self, custom_image=None, reaction_speed=None, prediction_factor=None):
        current_width, current_height = get_screen_dimensions()
        super().__init__(current_width * 3 // 4, current_height // 2, NEON_GREEN, custom_image)
        self.reaction_speed = reaction_speed if reaction_speed is not None else 0.1  # Velocidad original
        self.prediction_factor = prediction_factor if prediction_factor is not None else 0.4
        self.last_position_update = 0
        self.defensive_position = [current_width * 3 // 4, current_height // 2]
        
    def update(self, puck_pos, puck_velocity=None):
        current_width, current_height = get_screen_dimensions()
        self.prev_position = self.position.copy()
        
        # Movimiento simple de seguimiento, se reemplazará con RL
        if puck_pos:
            if puck_pos[0] > current_width // 2:  # Solo si la bola está en su mitad
                # Detectar si el puck está cerca de una esquina
                is_near_corner = (
                    (puck_pos[0] < self.radius*3 or puck_pos[0] > current_width - self.radius*3) and 
                    (puck_pos[1] < self.radius*3 or puck_pos[1] > current_height - self.radius*3)
                )
                
                if is_near_corner:
                    # Estrategia de esquina: moverse hacia el centro en lugar de hacia el puck
                    # Calcular vector desde la esquina hacia el centro de la mesa
                    target_x = current_width * 3 // 4  # Punto central de la mitad derecha
                    target_y = current_height // 2
                    
                    # Moverse hacia ese punto con suavidad
                    self.position[0] += (target_x - self.position[0]) * 0.15
                    self.position[1] += (target_y - self.position[1]) * 0.15
                else:
                    # Comportamiento normal con algo de inteligencia añadida
                    # Intentar interceptar el puck basado en su trayectoria, no solo su posición
                    # Calcular punto de interceptación estimado
                    intercept_x = min(max(puck_pos[0] + random.randint(-20, 20), current_width // 2 + self.radius), current_width - self.radius)
                    intercept_y = min(max(puck_pos[1] + random.randint(-20, 20), self.radius), current_height - self.radius)
                    
                    # Mover hacia el punto de interceptación con suavidad
                    self.position[0] += (intercept_x - self.position[0]) * self.reaction_speed
                    self.position[1] += (intercept_y - self.position[1]) * self.reaction_speed
                
                self.rect.center = self.position
                
                # Calcular velocidad basada en el cambio de posición
                self.velocity = [
                    self.position[0] - self.prev_position[0],
                    self.position[1] - self.prev_position[1]
                ]

class Puck(pygame.sprite.Sprite):
    """Clase para el disco (puck)"""
    
    def __init__(self, custom_image=None):
        super().__init__()
        # Calcular radio escalado
        scale_factor = min(get_screen_dimensions()[0]/800, get_screen_dimensions()[1]/500)
        self.radius = int(15 * scale_factor)
        
        if custom_image is not None:
            # Usar imagen personalizada
            self.image = custom_image
            # Asegurar que la imagen tenga el tamaño correcto
            if self.image.get_size() != (self.radius * 2, self.radius * 2):
                self.image = pygame.transform.smoothscale(self.image, (self.radius * 2, self.radius * 2))
        else:
            # Crear sprite por defecto
            self.image = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(self.image, NEON_BLUE, (self.radius, self.radius), self.radius)
            # Añadir brillo
            pygame.draw.circle(self.image, (255, 255, 255, 150), (self.radius, self.radius), self.radius // 2)
        
        current_width, current_height = get_screen_dimensions()
        self.rect = self.image.get_rect(center=(current_width // 2, current_height // 2))
        # Crear máscara circular para colisiones precisas
        self.mask = self._create_circular_mask()
        self.velocity = [random.choice([-2, 2]), random.choice([-2, 2])]
        self.position = [current_width // 2, current_height // 2]
        self.prev_position = self.position.copy()
        self.max_speed = MAX_PUCK_SPEED
        self.friction = FRICTION
    
    def _create_circular_mask(self):
        """Crea una máscara circular para colisiones precisas"""
        mask_surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(mask_surface, (255, 255, 255, 255), (self.radius, self.radius), self.radius)
        return pygame.mask.from_surface(mask_surface)
    
    def update(self):
        current_width, current_height = get_screen_dimensions()
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
            # Añadir un pequeño impulso para evitar que se pegue a la pared
            self.velocity[0] += 0.5
        elif new_x + self.radius > current_width:
            new_x = current_width - self.radius
            self.velocity[0] = -abs(self.velocity[0]) * COLLISION_ELASTICITY  # Asegurar rebote hacia adentro
            # Añadir un pequeño impulso para evitar que se pegue a la pared
            self.velocity[0] -= 0.5
            
        # Luego colisión vertical (arriba/abajo)
        if new_y - self.radius < 0:
            new_y = self.radius
            self.velocity[1] = abs(self.velocity[1]) * COLLISION_ELASTICITY  # Asegurar rebote hacia adentro
            # Añadir un pequeño impulso para evitar que se pegue a la pared
            self.velocity[1] += 0.5
        elif new_y + self.radius > current_height:
            new_y = current_height - self.radius
            self.velocity[1] = -abs(self.velocity[1]) * COLLISION_ELASTICITY  # Asegurar rebote hacia adentro
            # Añadir un pequeño impulso para evitar que se pegue a la pared
            self.velocity[1] -= 0.5
        
        # Verificación especial para las esquinas (añadir un rebote más fuerte)
        corner_radius = self.radius * 2  # Radio para detectar proximidad a esquinas
        corner_force = 1.5  # Fuerza de repulsión de las esquinas
        
        # Esquina superior izquierda
        if new_x < corner_radius and new_y < corner_radius:
            # Aplicar una fuerza que aleje el puck de la esquina
            corner_distance = math.sqrt(new_x**2 + new_y**2)
            if corner_distance < corner_radius:
                force_direction = normalize_vector([new_x, new_y])
                if force_direction[0] == 0 and force_direction[1] == 0:
                    force_direction = [1, 1]
                self.velocity[0] += force_direction[0] * corner_force
                self.velocity[1] += force_direction[1] * corner_force
            
        # Esquina superior derecha
        elif new_x > current_width - corner_radius and new_y < corner_radius:
            corner_distance = math.sqrt((current_width - new_x)**2 + new_y**2)
            if corner_distance < corner_radius:
                force_direction = normalize_vector([new_x - current_width, new_y])
                if force_direction[0] == 0 and force_direction[1] == 0:
                    force_direction = [-1, 1]
                self.velocity[0] += force_direction[0] * corner_force
                self.velocity[1] += force_direction[1] * corner_force
            
        # Esquina inferior izquierda
        elif new_x < corner_radius and new_y > current_height - corner_radius:
            corner_distance = math.sqrt(new_x**2 + (current_height - new_y)**2)
            if corner_distance < corner_radius:
                force_direction = normalize_vector([new_x, new_y - current_height])
                if force_direction[0] == 0 and force_direction[1] == 0:
                    force_direction = [1, -1]
                self.velocity[0] += force_direction[0] * corner_force
                self.velocity[1] += force_direction[1] * corner_force
            
        # Esquina inferior derecha
        elif new_x > current_width - corner_radius and new_y > current_height - corner_radius:
            corner_distance = math.sqrt((current_width - new_x)**2 + (current_height - new_y)**2)
            if corner_distance < corner_radius:
                force_direction = normalize_vector([new_x - current_width, new_y - current_height])
                if force_direction[0] == 0 and force_direction[1] == 0:
                    force_direction = [-1, -1]
                self.velocity[0] += force_direction[0] * corner_force
                self.velocity[1] += force_direction[1] * corner_force
        
        # Garantía absoluta de que el puck nunca sale del campo de juego
        new_x = max(self.radius, min(new_x, current_width - self.radius))
        new_y = max(self.radius, min(new_y, current_height - self.radius))
        
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
        current_width, current_height = get_screen_dimensions()
        self.position = [current_width // 2, current_height // 2]
        
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
        """Verificación avanzada de colisión con un mallet, con físicas mejoradas"""
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
            
            # Calcular distancia
            distance = vector_length([dx, dy])
            
            # Si la distancia es 0, usar un vector aleatorio pequeño
            if distance == 0:
                dx = 0.1
                dy = 0.1
                distance = vector_length([dx, dy])
            
            # Normalizar el vector de colisión
            collision_normal_x = dx / distance
            collision_normal_y = dy / distance
            
            # Separar los objetos para evitar superposición
            overlap = (mallet.radius + self.radius) - distance
            if overlap > 0:
                # Mover el puck fuera del mallet
                self.position[0] += collision_normal_x * (overlap + 1)
                self.position[1] += collision_normal_y * (overlap + 1)
            
            # Calcular velocidades relativas
            relative_velocity_x = self.velocity[0] - mallet.velocity[0]
            relative_velocity_y = self.velocity[1] - mallet.velocity[1]
            
            # Velocidad relativa en la dirección de la normal
            velocity_along_normal = (relative_velocity_x * collision_normal_x + 
                                   relative_velocity_y * collision_normal_y)
            
            # No hacer nada si las velocidades se están separando
            if velocity_along_normal > 0:
                return False
            
            # Calcular el impulso basado en la restitución (elasticidad)
            restitution = 1.2  # Un poco más que elástico para hacer el juego más dinámico
            impulse = 2 * velocity_along_normal / (1 + 0.5)  # Asumiendo que el mallet tiene más masa
            
            # Aplicar el impulso al puck
            self.velocity[0] -= impulse * collision_normal_x * restitution
            self.velocity[1] -= impulse * collision_normal_y * restitution
            
            # Añadir la velocidad del mallet para transferencia de momento
            mallet_influence = 0.3  # Qué tanto afecta la velocidad del mallet
            self.velocity[0] += mallet.velocity[0] * mallet_influence
            self.velocity[1] += mallet.velocity[1] * mallet_influence
            
            # Verificar si estamos cerca de un borde y ajustar para evitar que el puck salga
            current_width, current_height = get_screen_dimensions()
            margin = 30
            
            if self.position[0] - self.radius < margin:
                self.velocity[0] = abs(self.velocity[0])  # Forzar hacia la derecha
            elif self.position[0] + self.radius > current_width - margin:
                self.velocity[0] = -abs(self.velocity[0])  # Forzar hacia la izquierda
                
            if self.position[1] - self.radius < margin:
                self.velocity[1] = abs(self.velocity[1])  # Forzar hacia abajo
            elif self.position[1] + self.radius > current_height - margin:
                self.velocity[1] = -abs(self.velocity[1])  # Forzar hacia arriba
            
            # Limitar la velocidad máxima después de la colisión
            speed = vector_length(self.velocity)
            if speed > self.max_speed:
                normalized = normalize_vector(self.velocity)
                self.velocity[0] = normalized[0] * self.max_speed
                self.velocity[1] = normalized[1] * self.max_speed
            
            # Velocidad mínima para evitar que el puck se detenga
            min_speed = 2.0
            if speed < min_speed and speed > 0:
                normalized = normalize_vector(self.velocity)
                self.velocity[0] = normalized[0] * min_speed
                self.velocity[1] = normalized[1] * min_speed
            
            # Garantía absoluta de que el puck no salga del área jugable
            self.position[0] = max(self.radius, min(self.position[0], current_width - self.radius))
            self.position[1] = max(self.radius, min(self.position[1], current_height - self.radius))
            
            self.rect.center = self.position
            return True
                
        return False