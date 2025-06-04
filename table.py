import pygame
from constants import *

class Table:
    """Representa la mesa de air hockey"""
    
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.line_width = 4
        self.center_radius = 50
        self.goal_width = HEIGHT * GOAL_WIDTH_RATIO
        self.goal_y1 = HEIGHT * (1 - GOAL_WIDTH_RATIO) / 2
        self.goal_y2 = HEIGHT * (1 + GOAL_WIDTH_RATIO) / 2
        self.table_color = BLACK  # Default color, can be changed for themes
        
    def draw(self, screen):
        """Dibuja la mesa de air hockey con metas visibles"""
        # Fondo
        screen.fill(self.table_color)
        
        # Línea central
        pygame.draw.line(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), self.line_width)
        
        # Círculo central
        pygame.draw.circle(screen, WHITE, (WIDTH // 2, HEIGHT // 2), self.center_radius, self.line_width)
        
        # Porterías
        # Portería izquierda (jugador)
        goal_depth = 10  # Profundidad visual de la portería
        
        # Dibujar áreas de las porterías con transparencia
        s = pygame.Surface((goal_depth, self.goal_y2 - self.goal_y1), pygame.SRCALPHA)
        s.fill((255, 0, 0, 50))  # Rojo semi-transparente
        screen.blit(s, (0, self.goal_y1))
        
        # Borde de portería izquierda (jugador)
        pygame.draw.line(screen, NEON_RED, (0, self.goal_y1), (goal_depth, self.goal_y1), 3)  # Línea superior
        pygame.draw.line(screen, NEON_RED, (0, self.goal_y2), (goal_depth, self.goal_y2), 3)  # Línea inferior
        pygame.draw.line(screen, NEON_RED, (goal_depth, self.goal_y1), (goal_depth, self.goal_y2), 3)  # Línea derecha
        
        # Portería derecha (IA)
        s = pygame.Surface((goal_depth, self.goal_y2 - self.goal_y1), pygame.SRCALPHA)
        s.fill((0, 255, 0, 50))  # Verde semi-transparente
        screen.blit(s, (WIDTH - goal_depth, self.goal_y1))
        
        # Borde de portería derecha (IA)
        pygame.draw.line(screen, NEON_GREEN, (WIDTH, self.goal_y1), (WIDTH - goal_depth, self.goal_y1), 3)  # Línea superior
        pygame.draw.line(screen, NEON_GREEN, (WIDTH, self.goal_y2), (WIDTH - goal_depth, self.goal_y2), 3)  # Línea inferior
        pygame.draw.line(screen, NEON_GREEN, (WIDTH - goal_depth, self.goal_y1), (WIDTH - goal_depth, self.goal_y2), 3)  # Línea izquierda
        
        # Añadir efecto de luz a las porterías
        for i in range(3):
            alpha = 80 - i * 25
            if alpha > 0:
                # Portería izquierda
                s = pygame.Surface((goal_depth + i*2, (self.goal_y2 - self.goal_y1) + i*4), pygame.SRCALPHA)
                pygame.draw.rect(s, (255, 0, 0, alpha), (0, 0, goal_depth + i*2, (self.goal_y2 - self.goal_y1) + i*4), 1)
                screen.blit(s, (0 - i, self.goal_y1 - i*2))
                
                # Portería derecha
                s = pygame.Surface((goal_depth + i*2, (self.goal_y2 - self.goal_y1) + i*4), pygame.SRCALPHA)
                pygame.draw.rect(s, (0, 255, 0, alpha), (0, 0, goal_depth + i*2, (self.goal_y2 - self.goal_y1) + i*4), 1)
                screen.blit(s, (WIDTH - goal_depth - i, self.goal_y1 - i*2))
        
        # Bordes
        pygame.draw.rect(screen, WHITE, (0, 0, WIDTH, HEIGHT), self.line_width)
        
        # Efectos de luz en los bordes
        for i in range(5):
            alpha = 100 - i * 20
            if alpha > 0:
                s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.rect(s, (255, 255, 255, alpha), (i, i, WIDTH - 2*i, HEIGHT - 2*i), 1)
                screen.blit(s, (0, 0))
    
    def is_goal(self, puck):
        """Comprueba si hay un gol"""
        if puck.position[1] >= self.goal_y1 and puck.position[1] <= self.goal_y2:
            if puck.position[0] <= 0:
                return "ai"
            elif puck.position[0] >= WIDTH:
                return "player"
        return None