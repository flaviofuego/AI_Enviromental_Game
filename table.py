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
        
    def draw(self, screen, draw_background=True):
        """Dibuja la mesa de air hockey con metas visibles"""
        # Actualizar dimensiones en caso de que hayan cambiado
        current_width, current_height = get_screen_dimensions()
        
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
        else:
            # Dibujar portería por defecto
            s = pygame.Surface((goal_depth, int(self.goal_y2 - self.goal_y1)), pygame.SRCALPHA)
            s.fill((255, 0, 0, 50))
            screen.blit(s, (0, int(self.goal_y1)))
            
            pygame.draw.line(screen, NEON_RED, (0, int(self.goal_y1)), (goal_depth, int(self.goal_y1)), 3)
            pygame.draw.line(screen, NEON_RED, (0, int(self.goal_y2)), (goal_depth, int(self.goal_y2)), 3)
            pygame.draw.line(screen, NEON_RED, (goal_depth, int(self.goal_y1)), (goal_depth, int(self.goal_y2)), 3)
        
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
        else:
            # Dibujar portería por defecto
            s = pygame.Surface((goal_depth, int(self.goal_y2 - self.goal_y1)), pygame.SRCALPHA)
            s.fill((0, 255, 0, 50))
            screen.blit(s, (current_width - goal_depth, int(self.goal_y1)))
            
            pygame.draw.line(screen, NEON_GREEN, (current_width, int(self.goal_y1)), (current_width - goal_depth, int(self.goal_y1)), 3)
            pygame.draw.line(screen, NEON_GREEN, (current_width, int(self.goal_y2)), (current_width - goal_depth, int(self.goal_y2)), 3)
            pygame.draw.line(screen, NEON_GREEN, (current_width - goal_depth, int(self.goal_y1)), (current_width - goal_depth, int(self.goal_y2)), 3)
        
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
                
    def is_goal(self, puck):
        """Verifica si hay un gol con la nueva visualización de porterías"""
        current_width, current_height = get_screen_dimensions()
        
        # Gol en la portería izquierda (IA anota)
        if (puck.position[0] <= puck.radius and 
            puck.position[1] >= self.goal_y1 and 
            puck.position[1] <= self.goal_y2):
            return "ai"
            
        # Gol en la portería derecha (jugador anota)
        elif (puck.position[0] >= current_width - puck.radius and 
            puck.position[1] >= self.goal_y1 and 
            puck.position[1] <= self.goal_y2):
            return "player"
            
        return None