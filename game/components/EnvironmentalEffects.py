"""
Animated environmental background effects for the main menu.
Extracted from home.py for better component separation.

Performance: all surfaces are pre-created once in ``__init__`` and reused
each frame via ``fill()`` instead of allocating new ``pygame.Surface``
objects per particle per frame.
"""
import pygame
import math
import random


class EnvironmentalEffects:
    """Manages animated background particles and environmental effects."""

    def __init__(self, screen_width: int, screen_height: int, colors: dict):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.colors = colors

        self.particles = []
        self.pollution_particles = []
        self.heat_waves = []
        self.acid_rain = []
        self.melting_ice = []
        self.hope_sparkles = []
        self.aurora_strips = []

        self._create_particles()
        self._create_environmental_effects()

        # --- Pre-create reusable surfaces (performance) ---
        # Aurora: one wide strip per colour layer
        self._aurora_surf = pygame.Surface(
            (self.screen_width, 80), pygame.SRCALPHA
        ).convert_alpha()

        # Heat waves: one thin strip
        self._heat_surf = pygame.Surface(
            (self.screen_width, 4), pygame.SRCALPHA
        ).convert_alpha()

        # Pollution: keyed by size (max size = 8 → diameter = 16)
        self._pollution_surfs: dict[int, pygame.Surface] = {}
        for size in range(3, 9):
            s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA).convert_alpha()
            self._pollution_surfs[size] = s

        # Melting ice: cached at max expected size (ceil(20+10)=30 → 30×30)
        self._ice_surf = pygame.Surface((30, 30), pygame.SRCALPHA).convert_alpha()

        # Hope sparkles: keyed by particle size
        self._sparkle_surfs: dict[int, pygame.Surface] = {}
        for size in range(1, 4):
            s = pygame.Surface((size * 4, size * 4), pygame.SRCALPHA).convert_alpha()
            self._sparkle_surfs[size] = s

        # Generic particles: keyed by particle size
        self._particle_surfs: dict[int, pygame.Surface] = {}
        for size in range(1, 4):
            s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA).convert_alpha()
            self._particle_surfs[size] = s

        # Background temp surface (for alpha-blended background)
        self._bg_alpha_surf: pygame.Surface | None = None
        self._bg_alpha_value: int = -1

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _create_particles(self):
        for _ in range(20):
            self.particles.append({
                "x": pygame.math.Vector2(
                    random.randint(0, self.screen_width),
                    random.randint(0, self.screen_height),
                ),
                "vel": pygame.math.Vector2(
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5),
                ),
                "size": random.randint(1, 3),
                "color": random.choice([
                    self.colors["ice_blue"],
                    self.colors["warning_orange"],
                    (100, 100, 150),
                ]),
            })

    def _create_environmental_effects(self):
        for _ in range(15):
            self.pollution_particles.append({
                "x": random.randint(0, self.screen_width),
                "y": random.randint(0, self.screen_height),
                "vel_x": random.uniform(-0.3, 0.3),
                "vel_y": random.uniform(-0.8, -0.2),
                "size": random.randint(3, 8),
                "alpha": random.randint(30, 80),
                "life": random.randint(200, 400),
            })
        for _ in range(8):
            self.heat_waves.append({
                "y": random.randint(100, self.screen_height - 100),
                "amplitude": random.randint(10, 30),
                "frequency": random.uniform(0.02, 0.05),
                "speed": random.uniform(1, 3),
                "offset": random.uniform(0, 6.28),
            })
        for _ in range(25):
            self.acid_rain.append({
                "x": random.randint(-50, self.screen_width + 50),
                "y": random.randint(-100, -10),
                "vel_y": random.uniform(2, 5),
                "vel_x": random.uniform(-0.5, 0.5),
                "length": random.randint(5, 15),
            })
        for _ in range(6):
            self.melting_ice.append({
                "x": random.randint(50, self.screen_width - 50),
                "y": random.randint(50, 200),
                "drops": [],
                "last_drop": 0,
            })
        for _ in range(10):
            self.hope_sparkles.append({
                "x": random.randint(0, self.screen_width),
                "y": random.randint(0, self.screen_height),
                "vel_x": random.uniform(-0.5, 0.5),
                "vel_y": random.uniform(-0.5, 0.5),
                "size": random.randint(1, 3),
                "pulse": random.uniform(0, 6.28),
                "color_intensity": random.randint(100, 255),
            })
        for _ in range(5):
            self.aurora_strips.append({
                "points": [
                    (random.randint(0, self.screen_width), random.randint(0, 150))
                    for _ in range(6)
                ],
                "color_shift": random.uniform(0, 6.28),
                "flicker_intensity": random.uniform(0.3, 0.8),
            })

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt: float):
        for p in self.pollution_particles:
            p["x"] += p["vel_x"]
            p["y"] += p["vel_y"]
            p["life"] -= 1
            if p["life"] <= 0 or p["y"] < -20:
                p["x"] = random.randint(0, self.screen_width)
                p["y"] = self.screen_height + 20
                p["life"] = random.randint(200, 400)

        for wave in self.heat_waves:
            wave["offset"] += wave["speed"] * dt

        for drop in self.acid_rain:
            drop["x"] += drop["vel_x"]
            drop["y"] += drop["vel_y"]
            if drop["y"] > self.screen_height + 10:
                drop["x"] = random.randint(-50, self.screen_width + 50)
                drop["y"] = random.randint(-100, -10)

        for ice in self.melting_ice:
            ice["last_drop"] += dt
            if ice["last_drop"] > random.uniform(0.5, 2.0):
                ice["drops"].append({
                    "x": ice["x"] + random.randint(-5, 5),
                    "y": ice["y"],
                    "vel_y": random.uniform(1, 3),
                })
                ice["last_drop"] = 0
            for d in ice["drops"][:]:
                d["y"] += d["vel_y"]
                if d["y"] > self.screen_height:
                    ice["drops"].remove(d)

        for s in self.hope_sparkles:
            s["x"] += s["vel_x"]
            s["y"] += s["vel_y"]
            s["pulse"] += dt * 3
            if s["x"] < 0:
                s["x"] = self.screen_width
            elif s["x"] > self.screen_width:
                s["x"] = 0
            if s["y"] < 0:
                s["y"] = self.screen_height
            elif s["y"] > self.screen_height:
                s["y"] = 0

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface, animation_time: float,
             background_image=None, background_opacity: int = 200):
        # Background image — cache the alpha-applied copy
        if background_image:
            if (self._bg_alpha_surf is None
                    or self._bg_alpha_value != background_opacity):
                self._bg_alpha_surf = background_image.copy()
                self._bg_alpha_surf.set_alpha(background_opacity)
                self._bg_alpha_value = background_opacity
            surface.blit(self._bg_alpha_surf, (0, 0))

        self._draw_aurora(surface, animation_time)
        self._draw_heat_waves(surface, animation_time)
        self._draw_acid_rain(surface)
        self._draw_pollution(surface)
        self._draw_melting_ice(surface, animation_time)
        self._draw_hope_sparkles(surface, animation_time)
        self._draw_particles(surface, animation_time)

    # -- sub-draws --

    def _draw_aurora(self, surface, t):
        for aurora in self.aurora_strips:
            flicker = math.sin(t * 2 + aurora["color_shift"]) * aurora["flicker_intensity"]
            base_alpha = int(30 + 25 * flicker)
            if base_alpha <= 10:
                continue
            colors = [
                (255, 100, 100, base_alpha),
                (100, 255, 100, base_alpha),
                (150, 100, 255, base_alpha),
            ]
            for ci, color in enumerate(colors):
                a_surf = self._aurora_surf
                a_surf.fill((0, 0, 0, 0))
                points = []
                for j in range(len(aurora["points"])):
                    x = aurora["points"][j][0] + 20 * math.sin(t + j * 0.5)
                    y = aurora["points"][j][1] + ci * 15 + 10 * math.sin(t * 1.5 + j)
                    points.append((x, y))
                if len(points) > 2:
                    pygame.draw.lines(a_surf, color, False, points, 3)
                surface.blit(a_surf, (0, 0))

    def _draw_heat_waves(self, surface, t):
        for wave in self.heat_waves:
            points = []
            for x in range(0, self.screen_width, 10):
                y_off = wave["amplitude"] * math.sin(x * wave["frequency"] + wave["offset"])
                points.append((x, wave["y"] + y_off))
            if len(points) > 1:
                ws = self._heat_surf
                ws.fill((0, 0, 0, 0))
                for i in range(len(points) - 1):
                    alpha = int(40 + 20 * math.sin(t * 2 + points[i][0] * 0.01))
                    pygame.draw.line(ws, (255, 150, 50, alpha), points[i], points[i + 1], 2)
                surface.blit(ws, (0, 0))

    def _draw_acid_rain(self, surface):
        for drop in self.acid_rain:
            color = (200, 255, 100)
            sp = (int(drop["x"]), int(drop["y"]))
            ep = (int(drop["x"] + drop["vel_x"] * 2), int(drop["y"] + drop["length"]))
            if 0 <= sp[0] <= self.screen_width and 0 <= sp[1] <= self.screen_height:
                pygame.draw.line(surface, color, sp, ep, 1)

    def _draw_pollution(self, surface):
        for p in self.pollution_particles:
            if p["life"] > 0:
                size = p["size"]
                ps = self._pollution_surfs.get(size)
                if ps is None:
                    continue
                ps.fill((0, 0, 0, 0))
                alpha = min(p["alpha"], p["life"] // 2)
                pygame.draw.circle(ps, (60, 60, 60, alpha),
                                   (size, size), size)
                surface.blit(ps, (p["x"] - size, p["y"] - size))

    def _draw_melting_ice(self, surface, t):
        for ice in self.melting_ice:
            sz = int(20 + 10 * math.sin(t * 0.5))
            if sz > 30:
                sz = 30
            ic = (200, 230, 255, 150)
            is_ = self._ice_surf
            is_.fill((0, 0, 0, 0))
            pygame.draw.rect(is_, ic, (0, 0, sz, sz))
            surface.blit(is_, (ice["x"] - sz // 2, ice["y"] - sz // 2))
            for d in ice["drops"]:
                pygame.draw.circle(surface, (100, 150, 255), (int(d["x"]), int(d["y"])), 2)

    def _draw_hope_sparkles(self, surface, t):
        for s in self.hope_sparkles:
            pulse = (math.sin(s["pulse"]) + 1) * 0.5
            alpha = int(s["color_intensity"] * pulse)
            if alpha <= 20:
                continue
            size = s["size"]
            ss = self._sparkle_surfs.get(size)
            if ss is None:
                continue
            ss.fill((0, 0, 0, 0))
            color = (50, 255, 100, alpha)
            center = (size * 2, size * 2)
            points = []
            for i in range(8):
                angle = i * math.pi / 4
                radius = (size + 2) if i % 2 == 0 else (size // 2)
                x = center[0] + radius * math.cos(angle + s["pulse"])
                y = center[1] + radius * math.sin(angle + s["pulse"])
                points.append((x, y))
            if len(points) > 2:
                pygame.draw.polygon(ss, color, points)
            surface.blit(ss, (s["x"] - size * 2, s["y"] - size * 2))

    def _draw_particles(self, surface, t):
        for p in self.particles:
            p["x"] += p["vel"]
            px, py = p["x"].x, p["x"].y
            if px < 0:
                p["x"].x = self.screen_width
            elif px > self.screen_width:
                p["x"].x = 0
            if py < 0:
                p["x"].y = self.screen_height
            elif py > self.screen_height:
                p["x"].y = 0
            alpha = int(128 + 127 * math.sin(t * 2 + px * 0.01))
            color = (*p["color"], alpha)
            size = p["size"]
            ts = self._particle_surfs.get(size)
            if ts is None:
                continue
            ts.fill((0, 0, 0, 0))
            pygame.draw.circle(ts, color, (size, size), size)
            surface.blit(ts, (p["x"].x - size, p["x"].y - size))
