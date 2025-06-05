import pygame
import math
from constants import *

def draw_glow(surface, color, position, radius):
    """Dibuja un efecto de resplandor alrededor de un objeto - versión optimizada"""
    # Reducir capas de 4 a 2 para mejor rendimiento
    for i in range(2):
        alpha = 80 - i * 40  # Ajustar alpha para 2 capas
        if alpha > 0:
            s = pygame.Surface((radius*2 + i*6, radius*2 + i*6), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (radius + i*3, radius + i*3), radius + i*3)
            surface.blit(s, (position[0] - radius - i*3, position[1] - radius - i*3))

def calculate_vector(p1, p2):
    """Calcula el vector desde p1 hasta p2"""
    return [p2[0] - p1[0], p2[1] - p1[1]]

def vector_length(vector):
    """Calcula la longitud de un vector"""
    return math.sqrt(vector[0]**2 + vector[1]**2)

def normalize_vector(vector):
    """Normaliza un vector a longitud 1"""
    length = vector_length(vector)
    if length > 0:
        return [vector[0]/length, vector[1]/length]
    return [0, 0]

def dot_product(v1, v2):
    """Calcula el producto escalar de dos vectores"""
    return v1[0] * v2[0] + v1[1] * v2[1]

def line_circle_intersection(line_start, line_end, circle_center, circle_radius):
    """
    Determina si hay intersección entre un segmento de línea y un círculo
    Útil para detección de colisiones avanzada
    """
    line_dir = normalize_vector(calculate_vector(line_start, line_end))
    center_to_start = calculate_vector(circle_center, line_start)
    
    # Proyectar center_to_start sobre line_dir
    t = dot_product(center_to_start, line_dir)
    
    # Encontrar el punto más cercano en la línea al centro del círculo
    closest_point = [
        line_start[0] + line_dir[0] * t,
        line_start[1] + line_dir[1] * t
    ]
    
    # Si el punto más cercano está fuera del segmento de línea, encontrar el extremo más cercano
    line_length = vector_length(calculate_vector(line_start, line_end))
    if t < 0:
        closest_point = line_start
    elif t > line_length:
        closest_point = line_end
    
    # Verificar si la distancia es menor que el radio
    distance = vector_length(calculate_vector(circle_center, closest_point))
    return distance <= circle_radius