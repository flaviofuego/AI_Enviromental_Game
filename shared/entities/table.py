"""
Hockey table: draws the playing field, goals, and checks for goals/collisions.

Rendering is optimised via pre-cached surfaces so that ``draw()`` never
calls ``pygame.transform`` or creates temporary ``Surface`` objects.
"""
import logging
import pygame
from shared.config import GameConfig, PhysicsConfig, COLORS

logger = logging.getLogger(__name__)

# Aspect ratio shared by all goal-post sprites (width / height).
_GOAL_SPRITE_ASPECT = 520 / 949
# Fraction of the rendered goal width used for the physics hitbox depth.
_GOAL_HITBOX_DEPTH_RATIO = 0.3


class Table:
    """Represents the air-hockey table surface.

    **SRP contract**
    - ``set_goal_sprites`` / ``_compute_goal_cache``: prepare visual + hitbox data (run once).
    - ``draw``: blit pre-cached surfaces only (run every frame — zero allocations).
    - ``is_goal`` / ``check_goal_collision``: pure physics queries, no rendering.
    """

    def __init__(self, config: GameConfig = None, physics: PhysicsConfig = None):
        self.config = config or GameConfig()
        self.physics = physics or PhysicsConfig()
        self._update_dimensions()

        self.table_color = COLORS.BLACK

        # Raw source sprites (kept for invalidation / resize)
        self._goal_left_src: pygame.Surface | None = None
        self._goal_right_src: pygame.Surface | None = None

        # Pre-scaled, ready-to-blit goal surfaces + positions
        self._goal_left_scaled: pygame.Surface | None = None
        self._goal_right_scaled: pygame.Surface | None = None
        self._goal_left_pos: tuple[int, int] = (0, 0)
        self._goal_right_pos: tuple[int, int] = (0, 0)

        # Hitboxes (computed once, reused by physics)
        self.goal_left_hitbox: pygame.Rect | None = None
        self.goal_right_hitbox: pygame.Rect | None = None

        # Cached fallback surfaces (used when no sprite is set)
        self._fallback_left: pygame.Surface | None = None
        self._fallback_right: pygame.Surface | None = None
        self._fallback_left_pos: tuple[int, int] = (0, 0)
        self._fallback_right_pos: tuple[int, int] = (0, 0)

        # Cached decorative surfaces so ``draw()`` allocates nothing
        self._glow_surfaces: list[tuple[pygame.Surface, tuple[int, int]]] = []
        self._border_glow_surfaces: list[tuple[pygame.Surface, tuple[int, int]]] = []
        self._debug_left: pygame.Surface | None = None
        self._debug_right: pygame.Surface | None = None

        self.debug_mode = False
        self._cache_valid = False

    # ------------------------------------------------------------------
    # Dimension helpers
    # ------------------------------------------------------------------

    def _update_dimensions(self):
        c = self.config
        sf = c.scale_factor
        self.line_width = max(2, int(4 * sf))
        self.center_radius = int(50 * sf)
        self.goal_width = c.height * self.physics.goal_width_ratio
        self.goal_y1 = c.height * (1 - self.physics.goal_width_ratio) / 2
        self.goal_y2 = c.height * (1 + self.physics.goal_width_ratio) / 2

    # Backward-compatible read-only properties
    @property
    def goal_left_sprite(self):
        return self._goal_left_src

    @property
    def goal_right_sprite(self):
        return self._goal_right_src

    # ------------------------------------------------------------------
    # Sprite / cache management
    # ------------------------------------------------------------------

    def set_goal_sprites(self, left_sprite: pygame.Surface | None,
                         right_sprite: pygame.Surface | None) -> None:
        """Assign goal sprites and pre-compute all cached surfaces + hitboxes.

        Call this **once** after loading level assets.  ``draw()`` will
        then only blit the pre-computed surfaces.
        """
        self._goal_left_src = left_sprite
        self._goal_right_src = right_sprite
        self._rebuild_cache()
        logger.info(
            "Goal sprites set — left=%s  right=%s",
            f"{left_sprite.get_size()}" if left_sprite else "None",
            f"{right_sprite.get_size()}" if right_sprite else "None",
        )

    def invalidate_cache(self) -> None:
        """Force full cache rebuild on next ``draw()``."""
        self._cache_valid = False

    def _rebuild_cache(self) -> None:
        """Pre-compute every surface that ``draw()`` needs."""
        W, H = self.config.width, self.config.height
        gh = int(self.goal_y2 - self.goal_y1)
        goal_depth = max(5, int(10 * self.config.scale_factor))

        # --- Goal sprites (scaled) + hitboxes ---
        self._compute_goal_left_cache(W, gh, goal_depth)
        self._compute_goal_right_cache(W, gh, goal_depth)

        # --- Decorative glow for fallback goals ---
        self._glow_surfaces.clear()
        if self._goal_left_src is None and self._goal_right_src is None:
            for i in range(3):
                alpha = 80 - i * 25
                if alpha <= 0:
                    continue
                sw, sh = goal_depth + i * 2, gh + i * 4
                s_left = pygame.Surface((sw, sh), pygame.SRCALPHA)
                pygame.draw.rect(s_left, (255, 0, 0, alpha), (0, 0, sw, sh), 1)
                s_right = pygame.Surface((sw, sh), pygame.SRCALPHA)
                pygame.draw.rect(s_right, (0, 255, 0, alpha), (0, 0, sw, sh), 1)
                self._glow_surfaces.append((s_left, (-i, int(self.goal_y1) - i * 2)))
                self._glow_surfaces.append((s_right, (W - goal_depth - i, int(self.goal_y1) - i * 2)))

        # --- Border glow ---
        self._border_glow_surfaces.clear()
        for i in range(5):
            a = 100 - i * 20
            if a > 0:
                s = pygame.Surface((W, H), pygame.SRCALPHA)
                pygame.draw.rect(s, (255, 255, 255, a), (i, i, W - 2 * i, H - 2 * i), 1)
                self._border_glow_surfaces.append((s, (0, 0)))

        # --- Debug overlays ---
        self._build_debug_surfaces()

        self._cache_valid = True

    # --- per-goal cache builders ---

    def _compute_goal_left_cache(self, W: int, gh: int, goal_depth: int) -> None:
        if self._goal_left_src is not None:
            tw = int(gh * _GOAL_SPRITE_ASPECT)
            self._goal_left_scaled = pygame.transform.smoothscale(self._goal_left_src, (tw, gh))
            self._goal_left_pos = (0, int(self.goal_y1))
            gd = int(tw * _GOAL_HITBOX_DEPTH_RATIO)
            self.goal_left_hitbox = pygame.Rect(0, int(self.goal_y1), gd, gh)
        else:
            s = pygame.Surface((goal_depth, gh), pygame.SRCALPHA)
            s.fill((255, 0, 0, 50))
            # Burn lines directly onto the fallback surface
            pygame.draw.line(s, COLORS.NEON_RED, (0, 0), (goal_depth, 0), 3)
            pygame.draw.line(s, COLORS.NEON_RED, (0, gh - 1), (goal_depth, gh - 1), 3)
            pygame.draw.line(s, COLORS.NEON_RED, (goal_depth - 1, 0), (goal_depth - 1, gh), 3)
            self._fallback_left = s
            self._fallback_left_pos = (0, int(self.goal_y1))
            self._goal_left_scaled = None
            self.goal_left_hitbox = pygame.Rect(0, int(self.goal_y1), goal_depth, gh)

    def _compute_goal_right_cache(self, W: int, gh: int, goal_depth: int) -> None:
        if self._goal_right_src is not None:
            tw = int(gh * _GOAL_SPRITE_ASPECT)
            self._goal_right_scaled = pygame.transform.smoothscale(self._goal_right_src, (tw, gh))
            self._goal_right_pos = (W - tw, int(self.goal_y1))
            gd = int(tw * _GOAL_HITBOX_DEPTH_RATIO)
            self.goal_right_hitbox = pygame.Rect(W - gd, int(self.goal_y1), gd, gh)
        else:
            s = pygame.Surface((goal_depth, gh), pygame.SRCALPHA)
            s.fill((0, 255, 0, 50))
            pygame.draw.line(s, COLORS.NEON_GREEN, (0, 0), (goal_depth, 0), 3)
            pygame.draw.line(s, COLORS.NEON_GREEN, (0, gh - 1), (goal_depth, gh - 1), 3)
            pygame.draw.line(s, COLORS.NEON_GREEN, (0, 0), (0, gh), 3)
            self._fallback_right = s
            self._fallback_right_pos = (W - goal_depth, int(self.goal_y1))
            self._goal_right_scaled = None
            self.goal_right_hitbox = pygame.Rect(W - goal_depth, int(self.goal_y1), goal_depth, gh)

    def _build_debug_surfaces(self) -> None:
        if self.goal_left_hitbox:
            ds = pygame.Surface(self.goal_left_hitbox.size, pygame.SRCALPHA)
            ds.fill((255, 0, 0, 80))
            pygame.draw.rect(ds, COLORS.RED, (0, 0, *self.goal_left_hitbox.size), 2)
            self._debug_left = ds
        if self.goal_right_hitbox:
            ds = pygame.Surface(self.goal_right_hitbox.size, pygame.SRCALPHA)
            ds.fill((0, 255, 0, 80))
            pygame.draw.rect(ds, COLORS.GREEN, (0, 0, *self.goal_right_hitbox.size), 2)
            self._debug_right = ds

    # ------------------------------------------------------------------
    # Drawing (zero-allocation hot path)
    # ------------------------------------------------------------------

    def draw(self, screen, draw_background=True, debug_mode=False):
        """Render the table by blitting pre-cached surfaces only.

        On the first call (or after ``invalidate_cache()``), surfaces are
        rebuilt automatically so callers that never call ``set_goal_sprites``
        still get correct fallback rendering + hitboxes.
        """
        if not self._cache_valid:
            self._rebuild_cache()

        W, H = self.config.width, self.config.height
        self.debug_mode = debug_mode

        if draw_background:
            screen.fill(self.table_color)

        # Center line & circle
        pygame.draw.line(screen, COLORS.WHITE, (W // 2, 0), (W // 2, H), self.line_width)
        pygame.draw.circle(screen, COLORS.WHITE, (W // 2, H // 2), self.center_radius, self.line_width)

        # Goals — blit pre-cached surfaces (no transform, no Surface creation)
        if self._goal_left_scaled is not None:
            screen.blit(self._goal_left_scaled, self._goal_left_pos)
        elif self._fallback_left is not None:
            screen.blit(self._fallback_left, self._fallback_left_pos)

        if self._goal_right_scaled is not None:
            screen.blit(self._goal_right_scaled, self._goal_right_pos)
        elif self._fallback_right is not None:
            screen.blit(self._fallback_right, self._fallback_right_pos)

        # Glow effect on default goals (pre-cached surfaces)
        for surf, pos in self._glow_surfaces:
            screen.blit(surf, pos)

        # Border
        pygame.draw.rect(screen, COLORS.WHITE, (0, 0, W, H), self.line_width)

        # Border glow (pre-cached)
        for surf, pos in self._border_glow_surfaces:
            screen.blit(surf, pos)

        if self.debug_mode:
            self._draw_debug_hitboxes(screen)

    def _draw_debug_hitboxes(self, screen):
        if self._debug_left and self.goal_left_hitbox:
            screen.blit(self._debug_left, self.goal_left_hitbox.topleft)
        if self._debug_right and self.goal_right_hitbox:
            screen.blit(self._debug_right, self.goal_right_hitbox.topleft)

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
