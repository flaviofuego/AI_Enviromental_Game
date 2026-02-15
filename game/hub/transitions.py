"""
Screen transition effects for the game hub.
Ice-melt themed transitions between menu screens.
"""
import math
import random

import pygame


# Colores temáticos
_ICE_BLUE = (173, 216, 230)
_HEAT_RED = (220, 50, 50)
_FROST_WHITE = (240, 248, 255)
_COOL_BLUE = (100, 150, 200)


def ice_melt_transition(screen: pygame.Surface, fade_out: bool = True,
                        duration: float = 1.5) -> None:
    """Efecto de transición temático con derretimiento de hielo.

    Args:
        screen: Superficie de pygame donde se dibuja.
        fade_out: True = la imagen actual se desvanece, False = aparece.
        duration: Duración en segundos.
    """
    width, height = screen.get_size()
    clock = pygame.time.Clock()

    screenshot = screen.copy()

    # Partículas de hielo/calor
    particles = [
        {
            "x": random.randint(0, width),
            "y": random.randint(0, height),
            "size": random.randint(2, 6),
            "speed": random.uniform(1, 4),
            "angle": random.uniform(0, 2 * math.pi),
            "color": random.choice([_ICE_BLUE, _FROST_WHITE, _COOL_BLUE]),
            "life": random.uniform(0.5, 1.0),
        }
        for _ in range(150)
    ]

    total_frames = int(duration * 60)

    for frame in range(total_frames):
        dt = clock.tick(60) / 1000.0
        progress = frame / total_frames

        if fade_out:
            alpha = int(255 * (1 - progress))
            effect_alpha = int(255 * progress)
        else:
            alpha = int(255 * progress)
            effect_alpha = int(255 * (1 - progress))

        # Screenshot con transparencia
        temp_surface = screenshot.copy()
        temp_surface.set_alpha(alpha)
        screen.blit(temp_surface, (0, 0))

        # Superficie de efecto
        effect_surface = pygame.Surface((width, height), pygame.SRCALPHA)

        # Gradiente de fondo
        for y in range(0, height, 5):
            if fade_out:
                cr = int(_ICE_BLUE[0] * (1 - progress) + _HEAT_RED[0] * progress)
                cg = int(_ICE_BLUE[1] * (1 - progress) + _HEAT_RED[1] * progress)
                cb = int(_ICE_BLUE[2] * (1 - progress) + _HEAT_RED[2] * progress)
            else:
                cr = int(_HEAT_RED[0] * (1 - progress) + _ICE_BLUE[0] * progress)
                cg = int(_HEAT_RED[1] * (1 - progress) + _ICE_BLUE[1] * progress)
                cb = int(_HEAT_RED[2] * (1 - progress) + _ICE_BLUE[2] * progress)

            ratio = y / height
            cr = int(cr * (1 - ratio * 0.3))
            cg = int(cg * (1 - ratio * 0.3))
            cb = int(cb * (1 - ratio * 0.3))

            pygame.draw.line(effect_surface, (cr, cg, cb, effect_alpha),
                             (0, y), (width, y))

        # Partículas
        for p in particles:
            p["x"] += math.cos(p["angle"]) * p["speed"]
            p["y"] += math.sin(p["angle"]) * p["speed"] * 0.5
            if fade_out:
                p["y"] += progress * 3
            p["life"] -= dt * 0.5

            if p["y"] > height or p["life"] <= 0:
                p["y"] = -10
                p["x"] = random.randint(0, width)
                p["life"] = random.uniform(0.5, 1.0)

            pa = int(p["life"] * effect_alpha)
            if pa > 0:
                pygame.draw.circle(
                    effect_surface, (*p["color"], pa),
                    (int(p["x"]), int(p["y"])), p["size"],
                )
                halo_alpha = int(pa * 0.3)
                pygame.draw.circle(
                    effect_surface, (*_FROST_WHITE, halo_alpha),
                    (int(p["x"]), int(p["y"])), p["size"] + 2, 1,
                )

        # Onda de calor/frío
        if 0.3 < progress < 0.7:
            wave_i = math.sin((progress - 0.3) * math.pi / 0.4)
            for x in range(0, width, 20):
                wy = height // 2 + math.sin(x * 0.02 + frame * 0.1) * 50 * wave_i
                pygame.draw.circle(
                    effect_surface, (*_FROST_WHITE, int(30 * wave_i)),
                    (x, int(wy)), 15,
                )

        screen.blit(effect_surface, (0, 0))

        # Cristalización en bordes
        if fade_out and progress > 0.5:
            crystal_alpha = int((progress - 0.5) * 2 * 100)
            for i in range(int(progress * 50)):
                pygame.draw.line(screen, (*_FROST_WHITE, crystal_alpha),
                                 (0, i), (width, i))
                pygame.draw.line(screen, (*_FROST_WHITE, crystal_alpha),
                                 (0, height - i), (width, height - i))

        pygame.display.flip()

    if not fade_out:
        screen.blit(screenshot, (0, 0))
        pygame.display.flip()


def transition_effect(screen: pygame.Surface, fade_out: bool = True) -> None:
    """Transición corta (0.5 s) para cambios entre pantallas."""
    ice_melt_transition(screen, fade_out, duration=0.5)
