import pygame
import math

def draw_glow(surface, color, position, radius, glow_radius=4, alpha=128):
    """
    Draw a glowing effect around a point.
    
    Args:
        surface: Pygame surface to draw on
        color: RGB color tuple for the glow
        position: (x, y) position of the center
        radius: Base radius of the glow
        glow_radius: Additional radius for the glow effect
        alpha: Alpha value for the glow (0-255)
    """
    # Create a surface for the glow
    glow_surface = pygame.Surface((radius * 2 + glow_radius * 2, radius * 2 + glow_radius * 2), pygame.SRCALPHA)
    
    # Draw multiple circles with decreasing alpha for the glow effect
    for r in range(glow_radius, 0, -2):
        alpha_value = int(alpha * (r / glow_radius))
        pygame.draw.circle(
            glow_surface,
            (*color, alpha_value),
            (radius + glow_radius, radius + glow_radius),
            radius + r
        )
    
    # Draw the main circle
    pygame.draw.circle(
        glow_surface,
        (*color, 255),
        (radius + glow_radius, radius + glow_radius),
        radius
    )
    
    # Blit the glow surface onto the main surface
    surface.blit(
        glow_surface,
        (position[0] - radius - glow_radius, position[1] - radius - glow_radius)
    ) 