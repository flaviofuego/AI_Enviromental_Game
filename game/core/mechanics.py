"""
Level mechanics system for Air Hockey.
Each level has a unique environmental mechanic that affects gameplay.

Mechanics:
- Level 1: None (tutorial)
- Level 2: UV Zones — puck speeds up in radiation zones
- Level 3: Fog of War — reduced visibility with clear spots around mallet/puck
- Level 4: Shrinking Field — walls close in over time, goals push them back
- Level 5: Heat Waves — periodic friction reduction + visual distortion
"""
import math
import random
import pygame
from abc import ABC, abstractmethod
from shared.config import COLORS


class LevelMechanic(ABC):
    """Base class for level-specific mechanics."""

    def __init__(self, config, mechanic_params: dict):
        self.config = config
        self.params = mechanic_params
        self.active = True

    @abstractmethod
    def update(self, dt: float, puck, players: list, table):
        """Update mechanic state each frame."""

    @abstractmethod
    def draw(self, screen: pygame.Surface):
        """Draw mechanic visual effects."""

    def on_goal(self, scorer: str):
        """Called when a goal is scored. Override in subclasses."""

    def get_puck_speed_modifier(self) -> float:
        """Return puck speed multiplier (1.0 = normal)."""
        return 1.0

    def get_friction_modifier(self) -> float:
        """Return friction multiplier (1.0 = normal)."""
        return 1.0

    def get_field_bounds(self) -> tuple | None:
        """Return modified field bounds (left, top, right, bottom) or None for default."""
        return None


class NoneMechanic(LevelMechanic):
    """Level 1: No special mechanics (tutorial)."""

    def update(self, dt, puck, players, table):
        pass

    def draw(self, screen):
        pass


class UVZonesMechanic(LevelMechanic):
    """Level 2: UV radiation zones that speed up the puck."""

    def __init__(self, config, mechanic_params):
        super().__init__(config, mechanic_params)
        self.zone_count = mechanic_params.get("zone_count", 2)
        self.zone_radius = int(mechanic_params.get("zone_radius", 60) * config.scale_factor)
        self.speed_boost = mechanic_params.get("speed_boost", 1.3)
        self.move_speed = mechanic_params.get("zone_move_speed", 0.5) * config.scale_factor
        self.pulse_rate = mechanic_params.get("zone_pulse_rate", 2.0)

        W, H = config.width, config.height
        self.zones = []
        for _ in range(self.zone_count):
            x = random.uniform(W * 0.2, W * 0.8)
            y = random.uniform(H * 0.2, H * 0.8)
            dx = random.choice([-1, 1]) * self.move_speed
            dy = random.choice([-1, 1]) * self.move_speed * 0.5
            self.zones.append({"x": x, "y": y, "dx": dx, "dy": dy})

        self.time = 0.0
        self._puck_in_zone = False
        self._zone_surface = pygame.Surface((self.zone_radius * 2, self.zone_radius * 2), pygame.SRCALPHA)

    def update(self, dt, puck, players, table):
        self.time += dt
        W, H = self.config.width, self.config.height

        self._puck_in_zone = False
        for zone in self.zones:
            # Move zones
            zone["x"] += zone["dx"] * dt * 60
            zone["y"] += zone["dy"] * dt * 60

            # Bounce off edges
            if zone["x"] < self.zone_radius or zone["x"] > W - self.zone_radius:
                zone["dx"] *= -1
                zone["x"] = max(self.zone_radius, min(zone["x"], W - self.zone_radius))
            if zone["y"] < self.zone_radius or zone["y"] > H - self.zone_radius:
                zone["dy"] *= -1
                zone["y"] = max(self.zone_radius, min(zone["y"], H - self.zone_radius))

            # Check if puck is in zone
            dist = math.hypot(puck.position[0] - zone["x"], puck.position[1] - zone["y"])
            if dist < self.zone_radius:
                self._puck_in_zone = True
                # Apply speed boost to puck
                speed = math.hypot(puck.velocity[0], puck.velocity[1])
                if speed > 0.5:
                    boost = 1.0 + (self.speed_boost - 1.0) * dt * 2
                    puck.velocity[0] *= boost
                    puck.velocity[1] *= boost

    def draw(self, screen):
        pulse = 0.6 + 0.4 * math.sin(self.time * self.pulse_rate * math.pi * 2)

        for zone in self.zones:
            r = int(self.zone_radius * pulse)
            if r < 5:
                continue

            # Outer glow
            surf = pygame.Surface((r * 2 + 20, r * 2 + 20), pygame.SRCALPHA)
            alpha_outer = int(30 * pulse)
            pygame.draw.circle(surf, (180, 100, 255, alpha_outer), (r + 10, r + 10), r + 8)
            # Inner zone
            alpha_inner = int(50 * pulse)
            pygame.draw.circle(surf, (200, 130, 255, alpha_inner), (r + 10, r + 10), r)
            # Border ring
            pygame.draw.circle(surf, (220, 160, 255, int(100 * pulse)), (r + 10, r + 10), r, 2)

            screen.blit(surf, (int(zone["x"]) - r - 10, int(zone["y"]) - r - 10))

            # UV label
            if pulse > 0.8:
                font = pygame.font.Font(None, int(18 * self.config.scale_factor))
                label = font.render("UV", True, (200, 160, 255, int(180 * pulse)))
                screen.blit(label, (int(zone["x"]) - label.get_width() // 2,
                                    int(zone["y"]) - label.get_height() // 2))

    def get_puck_speed_modifier(self):
        return self.speed_boost if self._puck_in_zone else 1.0


class FogOfWarMechanic(LevelMechanic):
    """Level 3: Fog of war — reduced visibility around player mallet and puck."""

    def __init__(self, config, mechanic_params):
        super().__init__(config, mechanic_params)
        sf = config.scale_factor
        self.player_vision = int(mechanic_params.get("player_vision_radius", 120) * sf)
        self.puck_vision = int(mechanic_params.get("puck_vision_radius", 80) * sf)
        self.fog_opacity = mechanic_params.get("fog_opacity", 200)
        self.ai_sees_through = mechanic_params.get("ai_sees_through", True)

        # Pre-create fog surface
        self._fog_surface = pygame.Surface((config.width, config.height), pygame.SRCALPHA)
        self._clear_positions = []

    def update(self, dt, puck, players, table):
        # Store positions for drawing
        self._clear_positions = []
        if players:
            # Player 1 (human) always gets a clear spot
            self._clear_positions.append((players[0].position, self.player_vision))
        # Puck always visible
        self._clear_positions.append((puck.position, self.puck_vision))

    def draw(self, screen):
        W, H = self.config.width, self.config.height
        self._fog_surface.fill((30, 30, 40, self.fog_opacity))

        # Cut clear circles where player and puck are
        for pos, radius in self._clear_positions:
            # Gradient: fully clear in center, fading to fog at edge
            for r_frac in [1.0, 0.8, 0.6, 0.4, 0.2]:
                r = int(radius * r_frac)
                alpha = int(self.fog_opacity * (1 - r_frac))
                clear_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(clear_surf, (30, 30, 40, alpha), (r, r), r)
                x = int(pos[0]) - r
                y = int(pos[1]) - r
                self._fog_surface.blit(clear_surf, (x, y), special_flags=pygame.BLEND_RGBA_MIN)

        screen.blit(self._fog_surface, (0, 0))


class ShrinkingFieldMechanic(LevelMechanic):
    """Level 4: Field walls slowly close in. Player goals push back, AI goals shrink more."""

    def __init__(self, config, mechanic_params):
        super().__init__(config, mechanic_params)
        self.shrink_rate = mechanic_params.get("shrink_rate", 0.3) * config.scale_factor
        self.player_goal_expand = mechanic_params.get("player_goal_expand", 15) * config.scale_factor
        self.ai_goal_shrink = mechanic_params.get("ai_goal_shrink", 10) * config.scale_factor
        self.min_field_ratio = mechanic_params.get("min_field_ratio", 0.6)

        W, H = config.width, config.height
        self.original_bounds = (0, 0, W, H)
        self.max_shrink = (1.0 - self.min_field_ratio) * min(W, H) / 2

        self.shrink_offset_x = 0.0  # Shrink from left/right
        self.shrink_offset_y = 0.0  # Shrink from top/bottom
        self._wall_color = (100, 60, 30)
        self._warning_pulse = 0.0

    def update(self, dt, puck, players, table):
        # Walls close in over time
        self.shrink_offset_x = min(self.shrink_offset_x + self.shrink_rate * dt,
                                   self.max_shrink)
        self.shrink_offset_y = min(self.shrink_offset_y + self.shrink_rate * dt * 0.6,
                                   self.max_shrink * 0.6)
        self._warning_pulse += dt * 2

        # Constrain puck within shrunk bounds
        bounds = self.get_field_bounds()
        if bounds:
            left, top, right, bottom = bounds
            pr = puck.radius
            if puck.position[0] < left + pr:
                puck.position[0] = left + pr
                puck.velocity[0] = abs(puck.velocity[0]) * 0.8
            if puck.position[0] > right - pr:
                puck.position[0] = right - pr
                puck.velocity[0] = -abs(puck.velocity[0]) * 0.8
            if puck.position[1] < top + pr:
                puck.position[1] = top + pr
                puck.velocity[1] = abs(puck.velocity[1]) * 0.8
            if puck.position[1] > bottom - pr:
                puck.position[1] = bottom - pr
                puck.velocity[1] = -abs(puck.velocity[1]) * 0.8
            puck.rect.center = (int(puck.position[0]), int(puck.position[1]))

            # Constrain mallets too
            for player in players:
                mr = player.radius
                player.position[1] = max(top + mr, min(player.position[1], bottom - mr))
                player.rect.center = (int(player.position[0]), int(player.position[1]))

    def on_goal(self, scorer):
        if scorer == "player":
            # Player goal pushes walls back
            self.shrink_offset_x = max(0, self.shrink_offset_x - self.player_goal_expand)
            self.shrink_offset_y = max(0, self.shrink_offset_y - self.player_goal_expand * 0.6)
        elif scorer == "ai":
            # AI goal accelerates shrinking
            self.shrink_offset_x = min(self.shrink_offset_x + self.ai_goal_shrink, self.max_shrink)
            self.shrink_offset_y = min(self.shrink_offset_y + self.ai_goal_shrink * 0.6,
                                       self.max_shrink * 0.6)

    def draw(self, screen):
        bounds = self.get_field_bounds()
        if bounds is None:
            return
        left, top, right, bottom = bounds
        W, H = self.config.width, self.config.height

        pulse = 0.6 + 0.4 * abs(math.sin(self._warning_pulse))
        danger_ratio = self.shrink_offset_x / self.max_shrink if self.max_shrink > 0 else 0
        color_r = int(100 + 155 * danger_ratio * pulse)
        color_g = int(60 * (1 - danger_ratio))
        color_b = int(30 * (1 - danger_ratio))
        wall_color = (min(255, color_r), color_g, color_b)

        # Draw shrunk walls as filled rectangles on edges
        if left > 0:
            pygame.draw.rect(screen, wall_color, (0, 0, int(left), H))
        if top > 0:
            pygame.draw.rect(screen, wall_color, (0, 0, W, int(top)))
        if right < W:
            pygame.draw.rect(screen, wall_color, (int(right), 0, W - int(right), H))
        if bottom < H:
            pygame.draw.rect(screen, wall_color, (0, int(bottom), W, H - int(bottom)))

        # Inner border glow
        border_color = (min(255, color_r + 50), color_g + 20, color_b + 10, int(150 * pulse))
        border_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.rect(border_surf, border_color,
                         (int(left), int(top), int(right - left), int(bottom - top)), 3)
        screen.blit(border_surf, (0, 0))

    def get_field_bounds(self):
        left = self.shrink_offset_x
        top = self.shrink_offset_y
        right = self.config.width - self.shrink_offset_x
        bottom = self.config.height - self.shrink_offset_y
        if right - left < self.config.width * self.min_field_ratio:
            return None
        return (left, top, right, bottom)


class HeatWavesMechanic(LevelMechanic):
    """Level 5: Periodic heat waves reduce friction and add visual distortion."""

    def __init__(self, config, mechanic_params):
        super().__init__(config, mechanic_params)
        self.wave_interval = mechanic_params.get("wave_interval", 8.0)
        self.wave_duration = mechanic_params.get("wave_duration", 3.0)
        self.friction_reduction = mechanic_params.get("friction_reduction", 0.5)
        self.visual_distortion = mechanic_params.get("visual_distortion", True)
        self.trail_enabled = mechanic_params.get("trail_enabled", True)

        self.time_since_wave = 0.0
        self.wave_active = False
        self.wave_timer = 0.0
        self.total_time = 0.0

        # Heat trail
        self._trail_positions: list[tuple[float, float, float]] = []  # (x, y, age)
        self._heat_surface = pygame.Surface((config.width, config.height), pygame.SRCALPHA)

    def update(self, dt, puck, players, table):
        self.total_time += dt

        if self.wave_active:
            self.wave_timer += dt
            if self.wave_timer >= self.wave_duration:
                self.wave_active = False
                self.wave_timer = 0.0
                self.time_since_wave = 0.0
        else:
            self.time_since_wave += dt
            if self.time_since_wave >= self.wave_interval:
                self.wave_active = True
                self.wave_timer = 0.0

        # During heat wave: reduce friction (puck slides faster)
        if self.wave_active:
            # Reduce friction applied to puck — make it keep more speed
            speed = math.hypot(puck.velocity[0], puck.velocity[1])
            if speed > 0.1:
                # Counteract friction to simulate slipperiness
                boost = 1.0 + (1.0 - self.friction_reduction) * dt * 5
                puck.velocity[0] *= boost
                puck.velocity[1] *= boost

        # Trail positions
        if self.trail_enabled and self.wave_active:
            self._trail_positions.append((puck.position[0], puck.position[1], 0.0))

        # Age trail
        aged = []
        for x, y, age in self._trail_positions:
            new_age = age + dt
            if new_age < 1.5:
                aged.append((x, y, new_age))
        self._trail_positions = aged

    def draw(self, screen):
        # Draw heat trail
        if self.trail_enabled and self._trail_positions:
            for x, y, age in self._trail_positions:
                alpha = max(0, int(120 * (1 - age / 1.5)))
                r = max(2, int(8 * (1 - age / 1.5)))
                heat_color = (255, int(100 + 50 * (1 - age / 1.5)), 0, alpha)
                surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, heat_color, (r, r), r)
                screen.blit(surf, (int(x) - r, int(y) - r))

        # Heat wave visual overlay
        if self.wave_active and self.visual_distortion:
            W, H = self.config.width, self.config.height
            wave_progress = self.wave_timer / self.wave_duration
            intensity = math.sin(wave_progress * math.pi)  # Peaks in middle

            self._heat_surface.fill((0, 0, 0, 0))

            # Horizontal shimmer lines
            num_lines = 8
            for i in range(num_lines):
                y = int(H * (i + 0.5) / num_lines)
                offset = math.sin(self.total_time * 4 + i * 1.5) * 3 * intensity
                alpha = int(25 * intensity)
                color = (255, 140, 0, alpha)
                start_x = max(0, int(offset))
                pygame.draw.line(self._heat_surface, color,
                                 (start_x, y), (W, y), 1)

            # Overall orange tint
            tint_alpha = int(20 * intensity)
            tint_surf = pygame.Surface((W, H), pygame.SRCALPHA)
            tint_surf.fill((255, 100, 0, tint_alpha))
            screen.blit(tint_surf, (0, 0))
            screen.blit(self._heat_surface, (0, 0))

            # "HEAT WAVE" indicator
            if intensity > 0.3:
                font = pygame.font.Font(None, int(20 * self.config.scale_factor))
                pulse = 0.5 + 0.5 * math.sin(self.total_time * 6)
                warn_alpha = int(200 * pulse * intensity)
                txt = font.render("HEAT WAVE", True, (255, 80, 0))
                txt.set_alpha(warn_alpha)
                screen.blit(txt, (W // 2 - txt.get_width() // 2, 5))

    def get_friction_modifier(self):
        return self.friction_reduction if self.wave_active else 1.0


# ---- Factory ----

def create_mechanic(config, level_config: dict) -> LevelMechanic:
    """Create the appropriate mechanic for a given level config."""
    mechanics = level_config.get("mechanics", {})
    mtype = mechanics.get("type", "none")

    if mtype == "uv_zones":
        return UVZonesMechanic(config, mechanics)
    elif mtype == "fog_of_war":
        return FogOfWarMechanic(config, mechanics)
    elif mtype == "shrinking_field":
        return ShrinkingFieldMechanic(config, mechanics)
    elif mtype == "heat_waves":
        return HeatWavesMechanic(config, mechanics)
    else:
        return NoneMechanic(config, mechanics)
