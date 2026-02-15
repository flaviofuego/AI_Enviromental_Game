"""
Hockey table: draws the playing field, goals, and checks for goals/collisions.
"""
import pygame
from shared.config import GameConfig, PhysicsConfig, COLORS


class Table:
    """Represents the air-hockey table surface."""

    def __init__(self, config: GameConfig = None, physics: PhysicsConfig = None):
        self.config = config or GameConfig()
        self.physics = physics or PhysicsConfig()
        self._update_dimensions()

        self.table_color = COLORS.BLACK
        self.goal_left_sprite = None
        self.goal_right_sprite = None
        self.goal_left_hitbox = None
        self.goal_right_hitbox = None
        self.debug_mode = False

    def _update_dimensions(self):
        c = self.config
        sf = c.scale_factor
        self.line_width = max(2, int(4 * sf))
        self.center_radius = int(50 * sf)
        self.goal_width = c.height * self.physics.goal_width_ratio
        self.goal_y1 = c.height * (1 - self.physics.goal_width_ratio) / 2
        self.goal_y2 = c.height * (1 + self.physics.goal_width_ratio) / 2

    def set_goal_sprites(self, left_sprite, right_sprite):
        self.goal_left_sprite = left_sprite
        self.goal_right_sprite = right_sprite

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, screen, draw_background=True, debug_mode=False):
        W, H = self.config.width, self.config.height
        self.debug_mode = debug_mode

        if draw_background:
            screen.fill(self.table_color)

        # Center line & circle
        pygame.draw.line(screen, COLORS.WHITE, (W // 2, 0), (W // 2, H), self.line_width)
        pygame.draw.circle(screen, COLORS.WHITE, (W // 2, H // 2), self.center_radius, self.line_width)

        goal_depth = max(5, int(10 * self.config.scale_factor))

        # Left goal
        self._draw_goal_left(screen, W, H, goal_depth)
        # Right goal
        self._draw_goal_right(screen, W, H, goal_depth)

        # Glow effect on default goals
        if self.goal_left_sprite is None and self.goal_right_sprite is None:
            for i in range(3):
                alpha = 80 - i * 25
                if alpha <= 0:
                    continue
                gh = int(self.goal_y2 - self.goal_y1)
                s = pygame.Surface((goal_depth + i * 2, gh + i * 4), pygame.SRCALPHA)
                pygame.draw.rect(s, (255, 0, 0, alpha), (0, 0, s.get_width(), s.get_height()), 1)
                screen.blit(s, (-i, int(self.goal_y1) - i * 2))
                s2 = pygame.Surface((goal_depth + i * 2, gh + i * 4), pygame.SRCALPHA)
                pygame.draw.rect(s2, (0, 255, 0, alpha), (0, 0, s2.get_width(), s2.get_height()), 1)
                screen.blit(s2, (W - goal_depth - i, int(self.goal_y1) - i * 2))

        # Border
        pygame.draw.rect(screen, COLORS.WHITE, (0, 0, W, H), self.line_width)

        # Border glow
        for i in range(5):
            a = 100 - i * 20
            if a > 0:
                s = pygame.Surface((W, H), pygame.SRCALPHA)
                pygame.draw.rect(s, (255, 255, 255, a), (i, i, W - 2 * i, H - 2 * i), 1)
                screen.blit(s, (0, 0))

        if self.debug_mode:
            self._draw_debug_hitboxes(screen)

    def _draw_goal_left(self, screen, W, H, goal_depth):
        gh = int(self.goal_y2 - self.goal_y1)
        if self.goal_left_sprite is not None:
            ar = 520 / 949
            tw = int(gh * ar)
            sp = pygame.transform.smoothscale(self.goal_left_sprite, (tw, gh))
            screen.blit(sp, (0, int(self.goal_y1)))
            gd = int(tw * 0.3)
            self.goal_left_hitbox = pygame.Rect(0, int(self.goal_y1), gd, gh)
        else:
            s = pygame.Surface((goal_depth, gh), pygame.SRCALPHA)
            s.fill((255, 0, 0, 50))
            screen.blit(s, (0, int(self.goal_y1)))
            pygame.draw.line(screen, COLORS.NEON_RED, (0, int(self.goal_y1)), (goal_depth, int(self.goal_y1)), 3)
            pygame.draw.line(screen, COLORS.NEON_RED, (0, int(self.goal_y2)), (goal_depth, int(self.goal_y2)), 3)
            pygame.draw.line(screen, COLORS.NEON_RED, (goal_depth, int(self.goal_y1)), (goal_depth, int(self.goal_y2)), 3)
            self.goal_left_hitbox = pygame.Rect(0, int(self.goal_y1), goal_depth, gh)

    def _draw_goal_right(self, screen, W, H, goal_depth):
        gh = int(self.goal_y2 - self.goal_y1)
        if self.goal_right_sprite is not None:
            ar = 520 / 949
            tw = int(gh * ar)
            sp = pygame.transform.smoothscale(self.goal_right_sprite, (tw, gh))
            screen.blit(sp, (W - tw, int(self.goal_y1)))
            gd = int(tw * 0.3)
            self.goal_right_hitbox = pygame.Rect(W - gd, int(self.goal_y1), gd, gh)
        else:
            s = pygame.Surface((goal_depth, gh), pygame.SRCALPHA)
            s.fill((0, 255, 0, 50))
            screen.blit(s, (W - goal_depth, int(self.goal_y1)))
            pygame.draw.line(screen, COLORS.NEON_GREEN, (W, int(self.goal_y1)), (W - goal_depth, int(self.goal_y1)), 3)
            pygame.draw.line(screen, COLORS.NEON_GREEN, (W, int(self.goal_y2)), (W - goal_depth, int(self.goal_y2)), 3)
            pygame.draw.line(screen, COLORS.NEON_GREEN, (W - goal_depth, int(self.goal_y1)), (W - goal_depth, int(self.goal_y2)), 3)
            self.goal_right_hitbox = pygame.Rect(W - goal_depth, int(self.goal_y1), goal_depth, gh)

    def _draw_debug_hitboxes(self, screen):
        if self.goal_left_hitbox:
            ds = pygame.Surface((self.goal_left_hitbox.width, self.goal_left_hitbox.height), pygame.SRCALPHA)
            ds.fill((255, 0, 0, 80))
            screen.blit(ds, self.goal_left_hitbox.topleft)
            pygame.draw.rect(screen, COLORS.RED, self.goal_left_hitbox, 2)
        if self.goal_right_hitbox:
            ds = pygame.Surface((self.goal_right_hitbox.width, self.goal_right_hitbox.height), pygame.SRCALPHA)
            ds.fill((0, 255, 0, 80))
            screen.blit(ds, self.goal_right_hitbox.topleft)
            pygame.draw.rect(screen, COLORS.GREEN, self.goal_right_hitbox, 2)

    # ------------------------------------------------------------------
    # Goal detection
    # ------------------------------------------------------------------

    def is_goal(self, puck) -> str | None:
        """Return 'ai' if AI scored (left goal), 'player' if player scored (right goal)."""
        pr = pygame.Rect(
            puck.position[0] - puck.radius, puck.position[1] - puck.radius,
            puck.radius * 2, puck.radius * 2,
        )
        if self.goal_left_hitbox and pr.colliderect(self.goal_left_hitbox):
            hb = self.goal_left_hitbox
            if hb.left <= puck.position[0] <= hb.right and hb.top <= puck.position[1] <= hb.bottom:
                return "ai"
        if self.goal_right_hitbox and pr.colliderect(self.goal_right_hitbox):
            hb = self.goal_right_hitbox
            if hb.left <= puck.position[0] <= hb.right and hb.top <= puck.position[1] <= hb.bottom:
                return "player"
        return None

    # ------------------------------------------------------------------
    # Goal-structure collision (bounce off goal frame)
    # ------------------------------------------------------------------

    def check_goal_collision(self, puck) -> bool:
        W, H = self.config.width, self.config.height
        pr = pygame.Rect(
            puck.position[0] - puck.radius, puck.position[1] - puck.radius,
            puck.radius * 2, puck.radius * 2,
        )
        hit = False

        for hb, is_left in [(self.goal_left_hitbox, True), (self.goal_right_hitbox, False)]:
            if hb is None:
                continue
            tw = pygame.Rect(hb.left if not is_left else 0, int(self.goal_y1) - 5, hb.width, 5)
            bw = pygame.Rect(hb.left if not is_left else 0, int(self.goal_y2), hb.width, 5)
            if is_left:
                back = pygame.Rect(0, int(self.goal_y1), 5, int(self.goal_y2 - self.goal_y1))
            else:
                back = pygame.Rect(W - 5, int(self.goal_y1), 5, int(self.goal_y2 - self.goal_y1))

            if pr.colliderect(tw) and puck.velocity[1] > 0:
                puck.velocity[1] = -abs(puck.velocity[1]) * 0.8
                hit = True
            elif pr.colliderect(bw) and puck.velocity[1] < 0:
                puck.velocity[1] = abs(puck.velocity[1]) * 0.8
                hit = True
            elif pr.colliderect(back):
                if is_left and puck.velocity[0] < 0:
                    puck.velocity[0] = abs(puck.velocity[0]) * 0.8
                    hit = True
                elif not is_left and puck.velocity[0] > 0:
                    puck.velocity[0] = -abs(puck.velocity[0]) * 0.8
                    hit = True
        return hit
