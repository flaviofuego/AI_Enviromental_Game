"""
Puck entity with physics — used by both game and training.
"""
import math
import random
import pygame

from shared.config import GameConfig, PhysicsConfig, COLORS
from shared.physics import vector_length, normalize_vector, line_circle_intersection


class Puck(pygame.sprite.Sprite):
    """The hockey puck with wall collision, corner repulsion, and mallet collision."""

    def __init__(self, config: GameConfig = None, physics: PhysicsConfig = None,
                 custom_image=None):
        super().__init__()
        self.config = config or GameConfig()
        self.physics = physics or PhysicsConfig()
        self.radius = self.config.puck_radius

        # Visual
        if custom_image is not None:
            self.image = custom_image
            if self.image.get_size() != (self.radius * 2, self.radius * 2):
                self.image = pygame.transform.smoothscale(
                    self.image, (self.radius * 2, self.radius * 2)
                )
        else:
            self.image = pygame.Surface(
                (self.radius * 2, self.radius * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                self.image, COLORS.NEON_BLUE,
                (self.radius, self.radius), self.radius,
            )
            pygame.draw.circle(
                self.image, (255, 255, 255, 150),
                (self.radius, self.radius), self.radius // 2,
            )

        W, H = self.config.width, self.config.height
        self.rect = self.image.get_rect(center=(W // 2, H // 2))
        self.mask = self._create_circular_mask()
        self.velocity = [random.choice([-2, 2]), random.choice([-2, 2])]
        self.position = [float(W // 2), float(H // 2)]
        self.prev_position = self.position.copy()
        self.max_speed = self.physics.max_puck_speed
        self.friction = self.physics.friction

    def _create_circular_mask(self) -> pygame.mask.Mask:
        mask_surf = pygame.Surface(
            (self.radius * 2, self.radius * 2), pygame.SRCALPHA
        )
        pygame.draw.circle(
            mask_surf, (255, 255, 255, 255),
            (self.radius, self.radius), self.radius,
        )
        return pygame.mask.from_surface(mask_surf)

    # ------------------------------------------------------------------
    # Physics update
    # ------------------------------------------------------------------

    def update(self):
        W, H = self.config.width, self.config.height
        self.prev_position = self.position.copy()

        # Friction
        self.velocity[0] *= self.friction
        self.velocity[1] *= self.friction

        # Clamp speed
        speed = vector_length(self.velocity)
        if speed > self.max_speed:
            n = normalize_vector(self.velocity)
            self.velocity[0] = n[0] * self.max_speed
            self.velocity[1] = n[1] * self.max_speed

        new_x = self.position[0] + self.velocity[0]
        new_y = self.position[1] + self.velocity[1]
        elasticity = self.physics.collision_elasticity

        # Horizontal walls
        min_bounce = 1.0  # Minimum bounce speed to prevent sticking
        if new_x - self.radius < 0:
            new_x = self.radius
            self.velocity[0] = max(abs(self.velocity[0]) * elasticity, min_bounce)
        elif new_x + self.radius > W:
            new_x = W - self.radius
            self.velocity[0] = -max(abs(self.velocity[0]) * elasticity, min_bounce)

        # Vertical walls
        if new_y - self.radius < 0:
            new_y = self.radius
            self.velocity[1] = max(abs(self.velocity[1]) * elasticity, min_bounce)
        elif new_y + self.radius > H:
            new_y = H - self.radius
            self.velocity[1] = -max(abs(self.velocity[1]) * elasticity, min_bounce)

        # Corner repulsion
        corner_r = self.radius * 2
        corner_force = 1.5
        corners = [
            (0, 0),
            (W, 0),
            (0, H),
            (W, H),
        ]
        for cx, cy in corners:
            dx_c = new_x - cx
            dy_c = new_y - cy
            dist_c = math.hypot(dx_c, dy_c)
            if dist_c < corner_r and dist_c > 0:
                fd = normalize_vector([dx_c, dy_c])
                self.velocity[0] += fd[0] * corner_force
                self.velocity[1] += fd[1] * corner_force

        # Absolute bounds
        new_x = max(self.radius, min(new_x, W - self.radius))
        new_y = max(self.radius, min(new_y, H - self.radius))

        self.position = [new_x, new_y]
        self.rect.center = (int(new_x), int(new_y))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, scorer=None, zero_velocity=False):
        W, H = self.config.width, self.config.height
        self.position = [float(W // 2), float(H // 2)]

        if zero_velocity:
            self.velocity = [0.0, 0.0]
        elif scorer == "player":
            self.velocity = [random.uniform(-3, -1), random.uniform(-2, 2)]
        elif scorer == "ai":
            self.velocity = [random.uniform(1, 3), random.uniform(-2, 2)]
        else:
            self.velocity = [random.choice([-2, 2]), random.choice([-2, 2])]

        self.rect.center = (int(self.position[0]), int(self.position[1]))

    # ------------------------------------------------------------------
    # Mallet collision
    # ------------------------------------------------------------------

    def check_mallet_collision(self, mallet) -> bool:
        """Advanced mallet collision with impulse-based physics."""
        trajectory_hit = line_circle_intersection(
            self.prev_position, self.position,
            mallet.position, mallet.radius + self.radius,
        )
        standard_hit = pygame.sprite.collide_mask(self, mallet)

        if not (trajectory_hit or standard_hit):
            return False

        dx = self.position[0] - mallet.position[0]
        dy = self.position[1] - mallet.position[1]
        dist = vector_length([dx, dy])
        if dist == 0:
            dx, dy, dist = 0.1, 0.1, vector_length([0.1, 0.1])

        nx, ny = dx / dist, dy / dist

        # Separate
        overlap = (mallet.radius + self.radius) - dist
        if overlap > 0:
            self.position[0] += nx * (overlap + 1)
            self.position[1] += ny * (overlap + 1)

        # Relative velocity
        rvx = self.velocity[0] - mallet.velocity[0]
        rvy = self.velocity[1] - mallet.velocity[1]
        v_along_n = rvx * nx + rvy * ny
        if v_along_n > 0:
            # Puck already moving away — keep the separation but skip impulse
            W, H = self.config.width, self.config.height
            self.position[0] = max(self.radius, min(self.position[0], W - self.radius))
            self.position[1] = max(self.radius, min(self.position[1], H - self.radius))
            self.rect.center = (int(self.position[0]), int(self.position[1]))
            return False

        restitution = 1.2
        impulse = 2 * v_along_n / 1.5

        self.velocity[0] -= impulse * nx * restitution
        self.velocity[1] -= impulse * ny * restitution

        # Mallet momentum
        self.velocity[0] += mallet.velocity[0] * 0.3
        self.velocity[1] += mallet.velocity[1] * 0.3

        # Powerup strike boost: amplifies puck speed when this mallet hits
        strike_mult = getattr(mallet, "strike_multiplier", 1.0)
        if strike_mult != 1.0:
            self.velocity[0] *= strike_mult
            self.velocity[1] *= strike_mult

        # Edge guard
        W, H = self.config.width, self.config.height
        margin = 30
        if self.position[0] - self.radius < margin:
            self.velocity[0] = abs(self.velocity[0])
        elif self.position[0] + self.radius > W - margin:
            self.velocity[0] = -abs(self.velocity[0])
        if self.position[1] - self.radius < margin:
            self.velocity[1] = abs(self.velocity[1])
        elif self.position[1] + self.radius > H - margin:
            self.velocity[1] = -abs(self.velocity[1])

        # Speed clamp
        speed = vector_length(self.velocity)
        if speed > self.max_speed:
            n = normalize_vector(self.velocity)
            self.velocity[0] = n[0] * self.max_speed
            self.velocity[1] = n[1] * self.max_speed
        elif 0 < speed < 2.0:
            n = normalize_vector(self.velocity)
            self.velocity[0] = n[0] * 2.0
            self.velocity[1] = n[1] * 2.0

        self.position[0] = max(self.radius, min(self.position[0], W - self.radius))
        self.position[1] = max(self.radius, min(self.position[1], H - self.radius))
        self.rect.center = (int(self.position[0]), int(self.position[1]))
        return True
