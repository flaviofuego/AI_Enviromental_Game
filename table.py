import pygame
from constants import *

class Table:
    """Representa la mesa de air hockey"""
    
    def __init__(self):
        # Usar dimensiones dinámicas
        self.update_dimensions()
        # Color por defecto de la mesa (se puede cambiar externamente)
        self.table_color = BLACK
        # Sprites personalizados de porterías
        self.goal_left_sprite = None
        self.goal_right_sprite = None
        # Hitboxes de las porterías para detección precisa
        self.goal_left_hitbox = None
        self.goal_right_hitbox = None
        # Debug mode para mostrar hitboxes
        self.debug_mode = False
        
    def update_dimensions(self):
        """Actualiza las dimensiones de la mesa basándose en las constantes actuales"""
        current_width, current_height = get_screen_dimensions()
        self.width = current_width
        self.height = current_height
        self.line_width = max(2, int(4 * min(current_width/800, current_height/500)))
        self.center_radius = int(50 * min(current_width/800, current_height/500))
        self.goal_width = current_height * GOAL_WIDTH_RATIO
        self.goal_y1 = current_height * (1 - GOAL_WIDTH_RATIO) / 2
        self.goal_y2 = current_height * (1 + GOAL_WIDTH_RATIO) / 2
        
    def set_goal_sprites(self, left_sprite, right_sprite):
        """Establece los sprites personalizados para las porterías"""
        self.goal_left_sprite = left_sprite
        self.goal_right_sprite = right_sprite
        
    def draw(self, screen, draw_background=True, debug_mode=False):
        """Dibuja la mesa de air hockey con metas visibles"""
        # Actualizar dimensiones en caso de que hayan cambiado
        current_width, current_height = get_screen_dimensions()
        self.debug_mode = debug_mode
        
        # Fondo (solo si no hay fondo personalizado)
        if draw_background:
            screen.fill(self.table_color)
        
        # Línea central
        pygame.draw.line(screen, WHITE, (current_width // 2, 0), (current_width // 2, current_height), self.line_width)
        
        # Círculo central
        pygame.draw.circle(screen, WHITE, (current_width // 2, current_height // 2), self.center_radius, self.line_width)
        
        # Porterías
        goal_depth = max(5, int(10 * min(current_width/800, current_height/500)))
        
        # Dibujar portería izquierda
        if self.goal_left_sprite is not None:
            # Usar como base la relación de aspecto de la portería del nivel 1 (520x949)
            base_aspect_ratio = 520 / 949  # ~0.548
            
            # Calcular dimensiones objetivo basadas en la altura de la portería
            goal_height = int(self.goal_y2 - self.goal_y1)
            target_width = int(goal_height * base_aspect_ratio)
            
            # Escalar el sprite manteniendo la relación de aspecto base
            scaled_sprite = pygame.transform.smoothscale(self.goal_left_sprite, (target_width, goal_height))
            
            # Posicionar en el borde izquierdo
            screen.blit(scaled_sprite, (0, int(self.goal_y1)))
            
            # Crear hitbox para la portería izquierda (área interna de gol)
            goal_depth = int(target_width * 0.3)  # Profundidad del área de gol
            self.goal_left_hitbox = pygame.Rect(0, int(self.goal_y1), goal_depth, goal_height)
        else:
            # Dibujar portería por defecto
            goal_depth = max(5, int(10 * min(current_width/800, current_height/500)))
            s = pygame.Surface((goal_depth, int(self.goal_y2 - self.goal_y1)), pygame.SRCALPHA)
            s.fill((255, 0, 0, 50))
            screen.blit(s, (0, int(self.goal_y1)))
            
            pygame.draw.line(screen, NEON_RED, (0, int(self.goal_y1)), (goal_depth, int(self.goal_y1)), 3)
            pygame.draw.line(screen, NEON_RED, (0, int(self.goal_y2)), (goal_depth, int(self.goal_y2)), 3)
            pygame.draw.line(screen, NEON_RED, (goal_depth, int(self.goal_y1)), (goal_depth, int(self.goal_y2)), 3)
            
            # Crear hitbox para la portería izquierda por defecto
            self.goal_left_hitbox = pygame.Rect(0, int(self.goal_y1), goal_depth, int(self.goal_y2 - self.goal_y1))
        
        # Dibujar portería derecha
        if self.goal_right_sprite is not None:
            # Usar como base la relación de aspecto de la portería del nivel 1 (520x949)
            base_aspect_ratio = 520 / 949  # ~0.548
            
            # Calcular dimensiones objetivo basadas en la altura de la portería
            goal_height = int(self.goal_y2 - self.goal_y1)
            target_width = int(goal_height * base_aspect_ratio)
            
            # Escalar el sprite manteniendo la relación de aspecto base
            scaled_sprite = pygame.transform.smoothscale(self.goal_right_sprite, (target_width, goal_height))
            
            # Posicionar alineado al borde derecho
            x_position = current_width - target_width
            screen.blit(scaled_sprite, (x_position, int(self.goal_y1)))
            
            # Crear hitbox para la portería derecha (área interna de gol)
            goal_depth = int(target_width * 0.3)  # Profundidad del área de gol
            self.goal_right_hitbox = pygame.Rect(current_width - goal_depth, int(self.goal_y1), goal_depth, goal_height)
        else:
            # Dibujar portería por defecto
            goal_depth = max(5, int(10 * min(current_width/800, current_height/500)))
            s = pygame.Surface((goal_depth, int(self.goal_y2 - self.goal_y1)), pygame.SRCALPHA)
            s.fill((0, 255, 0, 50))
            screen.blit(s, (current_width - goal_depth, int(self.goal_y1)))
            
            pygame.draw.line(screen, NEON_GREEN, (current_width, int(self.goal_y1)), (current_width - goal_depth, int(self.goal_y1)), 3)
            pygame.draw.line(screen, NEON_GREEN, (current_width, int(self.goal_y2)), (current_width - goal_depth, int(self.goal_y2)), 3)
            pygame.draw.line(screen, NEON_GREEN, (current_width - goal_depth, int(self.goal_y1)), (current_width - goal_depth, int(self.goal_y2)), 3)
            
            # Crear hitbox para la portería derecha por defecto
            self.goal_right_hitbox = pygame.Rect(current_width - goal_depth, int(self.goal_y1), goal_depth, int(self.goal_y2 - self.goal_y1))
        
        # Añadir efecto de luz a las porterías (solo si no hay sprites personalizados)
        if self.goal_left_sprite is None and self.goal_right_sprite is None:
            for i in range(3):
                alpha = 80 - i * 25
                if alpha > 0:
                    # Portería izquierda
                    s = pygame.Surface((goal_depth + i*2, int(self.goal_y2 - self.goal_y1) + i*4), pygame.SRCALPHA)
                    pygame.draw.rect(s, (255, 0, 0, alpha), (0, 0, goal_depth + i*2, int(self.goal_y2 - self.goal_y1) + i*4), 1)
                    screen.blit(s, (0 - i, int(self.goal_y1) - i*2))
                    
                    # Portería derecha
                    s = pygame.Surface((goal_depth + i*2, int(self.goal_y2 - self.goal_y1) + i*4), pygame.SRCALPHA)
                    pygame.draw.rect(s, (0, 255, 0, alpha), (0, 0, goal_depth + i*2, int(self.goal_y2 - self.goal_y1) + i*4), 1)
                    screen.blit(s, (current_width - goal_depth - i, int(self.goal_y1) - i*2))
        
        # Bordes
        pygame.draw.rect(screen, WHITE, (0, 0, current_width, current_height), self.line_width)
        
        # Efectos de luz en los bordes
        for i in range(5):
            alpha = 100 - i * 20
            if alpha > 0:
                s = pygame.Surface((current_width, current_height), pygame.SRCALPHA)
                pygame.draw.rect(s, (255, 255, 255, alpha), (i, i, current_width - 2*i, current_height - 2*i), 1)
                screen.blit(s, (0, 0))
        
        # Mostrar hitboxes de debug si está activado
        if self.debug_mode:
            if self.goal_left_hitbox:
                # Hitbox de portería izquierda en rojo semi-transparente
                debug_surface = pygame.Surface((self.goal_left_hitbox.width, self.goal_left_hitbox.height), pygame.SRCALPHA)
                debug_surface.fill((255, 0, 0, 80))  # Rojo transparente
                screen.blit(debug_surface, (self.goal_left_hitbox.x, self.goal_left_hitbox.y))
                pygame.draw.rect(screen, (255, 0, 0), self.goal_left_hitbox, 2)  # Borde rojo
                
            if self.goal_right_hitbox:
                # Hitbox de portería derecha en verde semi-transparente
                debug_surface = pygame.Surface((self.goal_right_hitbox.width, self.goal_right_hitbox.height), pygame.SRCALPHA)
                debug_surface.fill((0, 255, 0, 80))  # Verde transparente
                screen.blit(debug_surface, (self.goal_right_hitbox.x, self.goal_right_hitbox.y))
                pygame.draw.rect(screen, (0, 255, 0), self.goal_right_hitbox, 2)  # Borde verde
                
    def is_goal(self, puck):
        """Verifica si hay un gol usando las hitboxes precisas de las porterías"""
        # Crear rect del puck para detección de colisión
        puck_rect = pygame.Rect(
            puck.position[0] - puck.radius,
            puck.position[1] - puck.radius,
            puck.radius * 2,
            puck.radius * 2
        )
        
        # Verificar gol en portería izquierda (IA anota)
        if self.goal_left_hitbox and puck_rect.colliderect(self.goal_left_hitbox):
            # Verificar que el centro del puck esté dentro de la hitbox para confirmar el gol
            if (puck.position[0] >= self.goal_left_hitbox.left and
                puck.position[0] <= self.goal_left_hitbox.right and
                puck.position[1] >= self.goal_left_hitbox.top and
                puck.position[1] <= self.goal_left_hitbox.bottom):
                return "ai"
        
        # Verificar gol en portería derecha (jugador anota)
        if self.goal_right_hitbox and puck_rect.colliderect(self.goal_right_hitbox):
            # Verificar que el centro del puck esté dentro de la hitbox para confirmar el gol
            if (puck.position[0] >= self.goal_right_hitbox.left and
                puck.position[0] <= self.goal_right_hitbox.right and
                puck.position[1] >= self.goal_right_hitbox.top and
                puck.position[1] <= self.goal_right_hitbox.bottom):
                return "player"
                
        return None
    
    def check_goal_collision(self, puck):
        """Verifica colisión del puck con las estructuras de las porterías para rebote"""
        current_width, current_height = get_screen_dimensions()
        
        # Crear rect del puck
        puck_rect = pygame.Rect(
            puck.position[0] - puck.radius,
            puck.position[1] - puck.radius,
            puck.radius * 2,
            puck.radius * 2
        )
        
        collision_occurred = False
        
        # Verificar colisión con portería izquierda
        if self.goal_left_hitbox:
            # Crear rects para las partes sólidas de la portería (paredes laterales)
            goal_top_wall = pygame.Rect(0, int(self.goal_y1) - 5, self.goal_left_hitbox.width, 5)
            goal_bottom_wall = pygame.Rect(0, int(self.goal_y2), self.goal_left_hitbox.width, 5)
            goal_back_wall = pygame.Rect(0, int(self.goal_y1), 5, int(self.goal_y2 - self.goal_y1))
            
            # Verificar colisiones con las paredes
            if puck_rect.colliderect(goal_top_wall) and puck.velocity[1] > 0:
                puck.velocity[1] = -abs(puck.velocity[1]) * 0.8  # Rebote hacia arriba
                collision_occurred = True
            elif puck_rect.colliderect(goal_bottom_wall) and puck.velocity[1] < 0:
                puck.velocity[1] = abs(puck.velocity[1]) * 0.8  # Rebote hacia abajo
                collision_occurred = True
            elif puck_rect.colliderect(goal_back_wall) and puck.velocity[0] < 0:
                puck.velocity[0] = abs(puck.velocity[0]) * 0.8  # Rebote hacia la derecha
                collision_occurred = True
        
        # Verificar colisión con portería derecha
        if self.goal_right_hitbox:
            # Crear rects para las partes sólidas de la portería (paredes laterales)
            goal_top_wall = pygame.Rect(self.goal_right_hitbox.left, int(self.goal_y1) - 5, self.goal_right_hitbox.width, 5)
            goal_bottom_wall = pygame.Rect(self.goal_right_hitbox.left, int(self.goal_y2), self.goal_right_hitbox.width, 5)
            goal_back_wall = pygame.Rect(current_width - 5, int(self.goal_y1), 5, int(self.goal_y2 - self.goal_y1))
            
            # Verificar colisiones con las paredes
            if puck_rect.colliderect(goal_top_wall) and puck.velocity[1] > 0:
                puck.velocity[1] = -abs(puck.velocity[1]) * 0.8  # Rebote hacia arriba
                collision_occurred = True
            elif puck_rect.colliderect(goal_bottom_wall) and puck.velocity[1] < 0:
                puck.velocity[1] = abs(puck.velocity[1]) * 0.8  # Rebote hacia abajo
                collision_occurred = True
            elif puck_rect.colliderect(goal_back_wall) and puck.velocity[0] > 0:
                puck.velocity[0] = -abs(puck.velocity[0]) * 0.8  # Rebote hacia la izquierda
                collision_occurred = True
        
        return collision_occurred