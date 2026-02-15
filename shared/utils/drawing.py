"""
Drawing helpers (glow effects, etc.) shared across the project.
"""
import pygame

# Module-level cache: (color, radius, glow_radius, alpha) -> Surface
_glow_cache: dict[tuple, pygame.Surface] = {}


def draw_glow(surface, color, position, radius, glow_radius=4, alpha=128):
    """
    Draw a glowing halo around a point.
    Uses a per-parameter cache so the surface is only built once per
    unique (color, radius, glow_radius, alpha) combination.

    Args:
        surface: Pygame surface to draw on.
        color: RGB tuple for the glow color.
        position: (x, y) center of the glow.
        radius: Base radius of the glow.
        glow_radius: Extra radius for the halo.
        alpha: Peak alpha value (0-255).
    """
    key = (color, radius, glow_radius, alpha)
    glow_surf = _glow_cache.get(key)

    if glow_surf is None:
        size = (radius + glow_radius) * 2
        glow_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        center = radius + glow_radius

        for r in range(glow_radius, 0, -2):
            a = int(alpha * (r / glow_radius))
            pygame.draw.circle(glow_surf, (*color, a), (center, center), radius + r)

        pygame.draw.circle(glow_surf, (*color, 255), (center, center), radius)
        _glow_cache[key] = glow_surf

    center = radius + glow_radius
    surface.blit(
        glow_surf,
        (position[0] - center, position[1] - center),
    )
