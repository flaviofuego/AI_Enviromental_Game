"""
game/entities/powerup_renderer.py
===================================
Visual representation layer for the powerup system (Section 4).

Design principles:
  - Single Responsibility: one class per visual concern.
  - Open/Closed: add a new powerup visual = add one private method; nothing else changes.
  - No game-logic mutation: this module only *reads* state and *draws* to screen.
  - No shared/ dependencies on pygame: all pygame code lives here, not in shared/.

Public API (used by game/core/renderer.py or game_engine.py):

    renderer = PowerUpRenderer()

    # --- Every frame ---
    renderer.update(dt)                             # advance animations
    renderer.draw_spheres(screen, manager, config)  # field sphere sprites
    renderer.draw_particles(screen)                 # burst particles
    renderer.draw_active_effects(                   # per-type mallet / field FX
        screen, manager, players, state, config
    )

    # --- On collection event (from manager.update() events list) ---
    renderer.emit_collection(event)  # event dict from PowerUpManager

Sub-renderers (instantiated internally):
  ParticleEmitter       – 15-particle radial bursts on collection / expiry
  SphereRenderer        – animated FieldSphere sprites (glow, pulse, blink)
  MalletEffectsRenderer – per-type continuous FX on mallets
  GoalEffectsRenderer   – shield arc on goal mouth
  FieldEffectsRenderer  – obstacle ice blocks on field
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import pygame

from game.components.IconRenderer import IconRenderer as _IconRenderer

# Module-level singleton so the cache persists across all renderer instances
_icon_renderer = _IconRenderer()

# ---------------------------------------------------------------------------
# Visual constants (tune everything here; keep logic files clean)
# ---------------------------------------------------------------------------

# --- Particle burst ---
PARTICLE_COUNT     = 15
PARTICLE_SPEED_MIN = 2.0
PARTICLE_SPEED_MAX = 5.0
PARTICLE_LIFETIME  = 0.4      # seconds
PARTICLE_RADIUS    = 3        # px

# --- Sphere ---
SPHERE_RADIUS      = 18       # logical radius (matches manager.FieldSphere.radius)
SPHERE_GLOW_LAYERS = 3        # concentric glow circles
SPHERE_GLOW_ALPHA  = 60       # per-layer glow alpha
SPHERE_PULSE_HZ    = 1.5      # pulse frequency (Hz)
SPHERE_PULSE_AMP   = 0.12     # ±12 % radius pulse

# --- Speed trail ---
TRAIL_LENGTH       = 5        # frames to keep
TRAIL_ALPHA_START  = 140
TRAIL_ALPHA_DECAY  = 25       # per frame

# --- Magnet ring ---
MAGNET_RING_EXTRA   = 8       # px beyond mallet radius
MAGNET_RING_PULSE   = 2.0     # Hz
MAGNET_RING_WIDTH   = 3       # px
MAGNET_RING_ALPHA   = 160

# --- Shield arc ---
SHIELD_ARC_WIDTH    = 6       # px
SHIELD_ARC_ALPHA    = 180
SHIELD_PULSE_HZ     = 1.8

# --- Duplication flash ---
DUP_FLASH_HZ        = 2.0     # flashes per second
DUP_FLASH_ALPHA     = 200

# --- Slow / paralysis halo ---
SLOW_HALO_ALPHA     = 120
PARA_ARC_COUNT      = 7       # lightning bolt segments per frame
PARA_ARC_INTERVAL   = 0.10    # seconds between arc regeneration

# --- Obstacles ---
OBS_SHINE_HZ        = 1.2
OBS_SHINE_ALPHA     = 80

# --- Invisibility ---
INVIS_SELF_ALPHA    = 150     # alpha shown on the owner's own screen
INVIS_OPP_ALPHA     = 20      # alpha shown on the opponent's screen

# ---------------------------------------------------------------------------
# Helper: quick surface with per-pixel alpha
# ---------------------------------------------------------------------------

def _circle_surface(radius: int, color: Tuple[int, int, int], alpha: int) -> pygame.Surface:
    """Return a square surface with a filled circle of *color* at *alpha*."""
    size = radius * 2
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(surf, (*color, alpha), (radius, radius), radius)
    return surf


def _ring_surface(
    radius: int,
    width: int,
    color: Tuple[int, int, int],
    alpha: int,
) -> pygame.Surface:
    """Return a surface with a drawn ring (hollow circle)."""
    size = (radius + width) * 2
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(surf, (*color, alpha), (radius + width, radius + width), radius, width)
    return surf


# ===========================================================================
# 1. Particle  (data only)
# ===========================================================================

class Particle:
    """Single airborne particle spawned on powerup collection."""

    __slots__ = ("x", "y", "vx", "vy", "color", "lifetime", "age", "radius")

    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        color: Tuple[int, int, int],
        lifetime: float = PARTICLE_LIFETIME,
        radius: int = PARTICLE_RADIUS,
    ) -> None:
        self.x        = x
        self.y        = y
        self.vx       = vx
        self.vy       = vy
        self.color    = color
        self.lifetime = lifetime
        self.age      = 0.0
        self.radius   = radius

    @property
    def is_dead(self) -> bool:
        return self.age >= self.lifetime

    @property
    def alpha(self) -> int:
        """Fade out linearly over lifetime."""
        frac = 1.0 - (self.age / self.lifetime)
        return int(255 * max(0.0, frac))


# ===========================================================================
# 2. ParticleEmitter
# ===========================================================================

class ParticleEmitter:
    """
    Manages all active particles.

    Usage:
        emitter.emit(x, y, count, color)   # radial burst
        emitter.update(dt)
        emitter.draw(screen)
    """

    def __init__(self) -> None:
        self._particles: List[Particle] = []

    # --- Public API -------------------------------------------------------

    def emit(
        self,
        x: float,
        y: float,
        count: int,
        color: Tuple[int, int, int],
        lifetime: float = PARTICLE_LIFETIME,
        speed_min: float = PARTICLE_SPEED_MIN,
        speed_max: float = PARTICLE_SPEED_MAX,
    ) -> None:
        """Fire *count* particles from *(x, y)* in random 360° directions."""
        for _ in range(count):
            angle = random.uniform(0.0, math.tau)
            speed = random.uniform(speed_min, speed_max)
            self._particles.append(
                Particle(
                    x       = x,
                    y       = y,
                    vx      = math.cos(angle) * speed,
                    vy      = math.sin(angle) * speed,
                    color   = color,
                    lifetime= lifetime,
                )
            )

    def update(self, dt: float) -> None:
        """Advance all particles and cull dead ones."""
        alive: List[Particle] = []
        for p in self._particles:
            p.age += dt
            if not p.is_dead:
                p.x += p.vx
                p.y += p.vy
                alive.append(p)
        self._particles = alive

    def draw(self, screen: pygame.Surface) -> None:
        """Draw all living particles with fade-out alpha."""
        for p in self._particles:
            surf = _circle_surface(p.radius, p.color, p.alpha)
            screen.blit(surf, (int(p.x) - p.radius, int(p.y) - p.radius))

    def __len__(self) -> int:
        return len(self._particles)


# ===========================================================================
# 3. SphereRenderer
# ===========================================================================

class SphereRenderer:
    """
    Draws FieldSphere objects (data from shared/powerups/manager.py) onto screen.

    Visual layers per sphere:
      1. Glow — soft concentric transparent circles
      2. Body — solid circle at the sphere's color
      3. Pulse — radius oscillates at SPHERE_PULSE_HZ
      4. Icon  — centered primitive icon (via IconRenderer)
      5. Alpha handled by the FieldSphere.alpha attribute (blink / fade)

    This renderer keeps an internal clock to drive pulsing animations
    independently of manager logic.
    """

    def __init__(self) -> None:
        self._t: float = 0.0

    def update(self, dt: float) -> None:
        self._t += dt

    def draw(
        self,
        screen: pygame.Surface,
        spheres: list,   # list[FieldSphere] — avoids circular import
    ) -> None:
        """Draw every sphere in *spheres* onto *screen*."""
        for sphere in spheres:
            self._draw_sphere(screen, sphere)

    # --- Private ----------------------------------------------------------

    def _draw_sphere(
        self,
        screen: pygame.Surface,
        sphere: Any,   # FieldSphere
    ) -> None:
        x = int(sphere.position[0])
        y = int(sphere.position[1])
        color = sphere.definition.color
        alpha = max(0, min(255, sphere.alpha))

        # Pulse: radius oscillates
        pulse = 1.0 + SPHERE_PULSE_AMP * math.sin(self._t * math.tau * SPHERE_PULSE_HZ)
        r = int(SPHERE_RADIUS * pulse)

        # --- Glow layers (outermost first) ---
        for layer in range(SPHERE_GLOW_LAYERS, 0, -1):
            glow_r   = r + layer * 5
            glow_a   = int(SPHERE_GLOW_ALPHA * (alpha / 255) / layer)
            glow_surf = _circle_surface(glow_r, color, glow_a)
            screen.blit(glow_surf, (x - glow_r, y - glow_r))

        # --- Body ---
        body_surf = _circle_surface(r, color, alpha)
        screen.blit(body_surf, (x - r, y - r))

        # --- Highlight (top-left white dot for 3D effect) ---
        hi_r = max(3, r // 4)
        hi_x = x - r // 3
        hi_y = y - r // 3
        hi_surf = _circle_surface(hi_r, (255, 255, 255), int(120 * alpha / 255))
        screen.blit(hi_surf, (hi_x - hi_r, hi_y - hi_r))

        # --- Icon (pygame primitive via IconRenderer) ---
        icon_size = max(10, r)
        icon_surf = _icon_renderer.get_powerup_surface(
            sphere.definition.icon, icon_size, (255, 255, 255), alpha
        )
        ir = icon_surf.get_rect(center=(x, y))
        screen.blit(icon_surf, ir)


# ===========================================================================
# 4. GoalEffectsRenderer
# ===========================================================================

class GoalEffectsRenderer:
    """
    Draws the shield arc over a player's goal mouth when shield is active.

    The arc is a semi-transparent orange band rendered at the left/right
    edge of the field depending on the player index.

    Player 0 defends the left goal (x ≈ 0).
    Player 1 defends the right goal (x ≈ config.width).
    """

    def __init__(self) -> None:
        self._t: float = 0.0

    def update(self, dt: float) -> None:
        self._t += dt

    def draw(
        self,
        screen: pygame.Surface,
        stacks: list,          # list[EffectStack]
        config: Any,           # GameConfig
    ) -> None:
        for player_idx, stack in enumerate(stacks):
            if stack.has_shield():
                self._draw_arc(screen, player_idx, config)

    # --- Private ----------------------------------------------------------

    def _draw_arc(
        self,
        screen: pygame.Surface,
        player_idx: int,
        config: Any,
    ) -> None:
        W = config.width
        H = config.height
        color = (255, 130, 60)

        # Pulse: alpha oscillates
        pulse_a = int(
            SHIELD_ARC_ALPHA
            * (0.7 + 0.3 * math.sin(self._t * math.tau * SHIELD_PULSE_HZ))
        )

        # Goal mouth position
        # Player 0 → left side (x=0), Player 1 → right side (x=W)
        goal_h = getattr(config, "goal_height", H * 0.4)
        top_y  = int((H - goal_h) / 2)
        bot_y  = int((H + goal_h) / 2)
        arc_w  = SHIELD_ARC_WIDTH

        if player_idx == 0:
            x = 0
            rect = pygame.Rect(x, top_y, arc_w * 4, bot_y - top_y)
        else:
            x = W - arc_w * 4
            rect = pygame.Rect(x, top_y, arc_w * 4, bot_y - top_y)

        shield_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        shield_surf.fill((*color, pulse_a))
        screen.blit(shield_surf, rect.topleft)

        # Border lines
        border_color = (*color, min(255, pulse_a + 60))
        pygame.draw.rect(screen, border_color, rect, SHIELD_ARC_WIDTH)


# ===========================================================================
# 5. FieldEffectsRenderer
# ===========================================================================

class FieldEffectsRenderer:
    """
    Draws ice-block obstacles placed on the field by the Obstacle powerup (phase 6).

    Reads `state.obstacles` (list of dicts written by phase6_obstacle.py).
    Handles fading-out obstacles and the "shine" pulse animation.
    Also implements the anti-stuck mechanism cleanup (visual only — engine handles physics).
    """

    FADE_SPEED = 510  # alpha units per second during fade-out (0→255 in 0.5 s)

    def __init__(self) -> None:
        self._t: float = 0.0

    def update(self, dt: float, state: Any) -> None:
        self._t += dt
        obstacles = getattr(state, "obstacles", None)
        if not obstacles:
            return
        to_remove = []
        for obs in obstacles:
            if obs.get("fading"):
                obs["alpha"] = max(0, obs["alpha"] - self.FADE_SPEED * dt)
                if obs["alpha"] <= 0:
                    to_remove.append(obs)
        for obs in to_remove:
            obstacles.remove(obs)

    def draw(self, screen: pygame.Surface, state: Any) -> None:
        obstacles = getattr(state, "obstacles", None)
        if not obstacles:
            return
        for obs in obstacles:
            self._draw_obstacle(screen, obs)

    # --- Private ----------------------------------------------------------

    def _draw_obstacle(self, screen: pygame.Surface, obs: dict) -> None:
        x      = int(obs["x"])
        y      = int(obs["y"])
        r      = obs["radius"]
        alpha  = int(obs.get("alpha", 255))
        color  = (180, 230, 255)    # ice blue

        # Shine pulse
        shine = int(
            OBS_SHINE_ALPHA
            * abs(math.sin(self._t * math.tau * OBS_SHINE_HZ))
        )

        # Body (semi-transparent ice)
        body_surf = _circle_surface(r, color, int(alpha * 0.75))
        screen.blit(body_surf, (x - r, y - r))

        # Shine overlay
        shine_surf = _circle_surface(r // 2, (255, 255, 255), int(shine * alpha / 255))
        screen.blit(shine_surf, (x - r // 4, y - r // 2))

        # Outline
        outline_a = min(255, alpha)
        if outline_a > 10:
            pygame.draw.circle(screen, (*color, outline_a), (x, y), r, 2)


# ===========================================================================
# 6. MalletEffectsRenderer
# ===========================================================================

class MalletEffectsRenderer:
    """
    Draws continuous per-type visual effects on mallets.

    Supported effects (one private method per powerup type):
      _draw_speed_trail      – cyan trailing ghost circles (speed_boost)
      _draw_magnet_ring      – pulsing purple ring (magnet)
      _draw_duplication_flash– gold flash (duplication)
      _draw_slow_halo        – grey smoke halo (slow_opponent, on opponent)
      _draw_paralyze_arcs    – yellow lightning bolts (paralyze, on opponent)
      _draw_invisibility     – alpha management (invisibility)

    Invisibility alpha is applied externally by the game renderer
    (it needs to control the mallet sprite's alpha channel).
    This renderer only draws the auxiliary visual cue for the *owner*.
    """

    def __init__(self) -> None:
        self._t: float = 0.0
        # Per-player mallet position history for the speed trail
        self._trails: Dict[int, Deque[Tuple[float, float]]] = {
            0: deque(maxlen=TRAIL_LENGTH),
            1: deque(maxlen=TRAIL_LENGTH),
        }
        # Per-player paralysis arc state
        self._para_arcs: Dict[int, List[Tuple[float, float]]] = {0: [], 1: []}
        self._para_timers: Dict[int, float] = {0: 0.0, 1: 0.0}
        self._font: Optional[pygame.font.Font] = None

    def _get_font(self) -> pygame.font.Font:
        if self._font is None:
            self._font = pygame.font.SysFont("segoe ui emoji", 14)
        return self._font

    # --- Main update/draw -------------------------------------------------

    def update(self, dt: float, players: list, stacks: list) -> None:
        """Advance internal animation state."""
        self._t += dt
        # Record trail positions
        for i, player in enumerate(players):
            if i in self._trails:
                pos = (float(player.position[0]), float(player.position[1]))
                self._trails[i].append(pos)
        # Regenerate paralysis arcs periodically
        for i, stack in enumerate(stacks):
            if stack.is_paralyzed():
                self._para_timers[i] += dt
                if self._para_timers[i] >= PARA_ARC_INTERVAL:
                    self._para_timers[i] = 0.0
                    self._para_arcs[i] = self._gen_arcs(players[i])
            else:
                self._para_arcs[i] = []
                self._para_timers[i] = 0.0

    def draw(
        self,
        screen: pygame.Surface,
        players: list,
        stacks: list,
    ) -> None:
        """Draw all mallet-based visual effects."""
        for i, (player, stack) in enumerate(zip(players, stacks)):
            # Effects that decorate the player who collected the powerup
            if stack.has("speed_boost"):
                self._draw_speed_trail(screen, i, (0, 200, 255))
            if stack.has("magnet"):
                self._draw_magnet_ring(screen, player)
            if stack.has("duplication"):
                self._draw_duplication_flash(screen, player)
            if stack.is_invisible():
                self._draw_invisibility_cue(screen, player)

        # Effects that decorate the *opponent* (drawn from opponent's stack perspective)
        # We iterate players and check if the *other* player has the effect targeting them
        for i, (player, stack) in enumerate(zip(players, stacks)):
            if stack.is_paralyzed():
                self._draw_paralyze_arcs(screen, player, i)
            # Slow halo: check if opponent applied slow to this player
            # EffectStack tracks effects targeting this player index
            if stack.has("slow_opponent"):
                self._draw_slow_halo(screen, player, (130, 115, 100))

    # --- Per-type private renderers ---------------------------------------

    def _draw_speed_trail(
        self,
        screen: pygame.Surface,
        player_idx: int,
        color: Tuple[int, int, int],
    ) -> None:
        """Render a fading cyan trail behind the mallet."""
        trail = self._trails.get(player_idx, deque())
        trail_list = list(trail)
        for frame_offset, (px, py) in enumerate(reversed(trail_list)):
            alpha = max(0, TRAIL_ALPHA_START - frame_offset * TRAIL_ALPHA_DECAY)
            radius = max(4, SPHERE_RADIUS // 2 - frame_offset)
            trail_surf = _circle_surface(radius, color, alpha)
            screen.blit(trail_surf, (int(px) - radius, int(py) - radius))

    def _draw_magnet_ring(
        self,
        screen: pygame.Surface,
        player: Any,
    ) -> None:
        """Pulsing purple ring with radius = mallet_radius + MAGNET_RING_EXTRA."""
        base_r = getattr(player, "radius", 28)
        pulse  = 0.9 + 0.1 * math.sin(self._t * math.tau * MAGNET_RING_PULSE)
        ring_r = int((base_r + MAGNET_RING_EXTRA) * pulse)
        color  = (170, 50, 220)
        surf   = _ring_surface(ring_r, MAGNET_RING_WIDTH, color, MAGNET_RING_ALPHA)
        cx = int(player.position[0])
        cy = int(player.position[1])
        offset = ring_r + MAGNET_RING_WIDTH
        screen.blit(surf, (cx - offset, cy - offset))

    def _draw_duplication_flash(
        self,
        screen: pygame.Surface,
        player: Any,
    ) -> None:
        """Gold intermittent flash over the mallet."""
        alpha = int(DUP_FLASH_ALPHA * abs(math.sin(self._t * math.tau * DUP_FLASH_HZ)))
        color = (255, 210, 0)
        r     = getattr(player, "radius", 28)
        surf  = _circle_surface(r, color, alpha)
        cx = int(player.position[0])
        cy = int(player.position[1])
        screen.blit(surf, (cx - r, cy - r))

    def _draw_slow_halo(
        self,
        screen: pygame.Surface,
        player: Any,
        color: Tuple[int, int, int],
    ) -> None:
        """Smoky grey halo around a slowed mallet."""
        r    = getattr(player, "radius", 28) + 10
        surf = _circle_surface(r, color, SLOW_HALO_ALPHA)
        cx = int(player.position[0])
        cy = int(player.position[1])
        screen.blit(surf, (cx - r, cy - r))

    def _draw_paralyze_arcs(
        self,
        screen: pygame.Surface,
        player: Any,
        player_idx: int,
    ) -> None:
        """Procedural lightning arcs around the paralyzed mallet."""
        arcs = self._para_arcs.get(player_idx, [])
        if not arcs:
            return
        color = (255, 240, 0)
        for (ox, oy) in arcs:
            cx = int(player.position[0])
            cy = int(player.position[1])
            end_x = cx + int(ox)
            end_y = cy + int(oy)
            pygame.draw.line(screen, color, (cx, cy), (end_x, end_y), 2)

    def _draw_invisibility_cue(
        self,
        screen: pygame.Surface,
        player: Any,
    ) -> None:
        """
        Semi-transparent white shimmer so the owner knows they are invisible.
        A subtle breathing pulse reminds the player of the active effect.
        """
        alpha = int(
            INVIS_SELF_ALPHA
            * (0.7 + 0.3 * math.sin(self._t * math.tau * 1.5))
        )
        r     = getattr(player, "radius", 28)
        surf  = _circle_surface(r, (200, 200, 255), alpha)
        cx = int(player.position[0])
        cy = int(player.position[1])
        screen.blit(surf, (cx - r, cy - r))

    # --- Arc generation helper -------------------------------------------

    @staticmethod
    def _gen_arcs(player: Any) -> List[Tuple[float, float]]:
        """
        Generate random lightning arc offsets (relative to mallet center).
        Returns a list of (dx, dy) endpoints for line drawing.
        """
        r      = getattr(player, "radius", 28) + 15
        result = []
        for _ in range(PARA_ARC_COUNT):
            angle = random.uniform(0, math.tau)
            dist  = random.uniform(r * 0.5, r * 1.2)
            result.append((math.cos(angle) * dist, math.sin(angle) * dist))
        return result

    # --- Utility ----------------------------------------------------------

    def get_invisibility_alpha(
        self,
        player_idx: int,
        stacks: list,
        is_own_screen: bool,
    ) -> int:
        """
        Return the alpha value for the mallet sprite at *player_idx*.

        is_own_screen = True  → this is the owner's perspective → 150
        is_own_screen = False → opponent's view                 → 20
        Returns 255 if invisibility is not active.
        """
        if stacks[player_idx].is_invisible():
            return INVIS_SELF_ALPHA if is_own_screen else INVIS_OPP_ALPHA
        return 255



# ===========================================================================
# 7. PowerUpRenderer  (Façade)
# ===========================================================================

class PowerUpRenderer:
    """
    Façade that combines all sub-renderers into a single cohesive API.

    Intended usage inside the game loop::

        # Instantiate once
        pu_renderer = PowerUpRenderer()

        # Every frame
        events = powerup_manager.update(dt, players, puck, state)
        pu_renderer.update(dt, manager, players, state)

        # React to manager events (collection, expiry)
        for ev in events:
            pu_renderer.on_manager_event(ev)

        # Draw layers in z-order:
        pu_renderer.draw_spheres(screen, manager)           # field spheres
        pu_renderer.draw_active_effects(screen, manager, players, state, config)
        pu_renderer.draw_particles(screen)                  # always on top
    """

    def __init__(self) -> None:
        self._particles    = ParticleEmitter()
        self._spheres      = SphereRenderer()
        self._goals        = GoalEffectsRenderer()
        self._field_obs    = FieldEffectsRenderer()
        self._mallet_fx    = MalletEffectsRenderer()

    # --- Per-frame update -------------------------------------------------

    def update(
        self,
        dt: float,
        manager: Any,        # PowerUpManager
        players: list,
        state: Any,
    ) -> None:
        """Advance all sub-renderer animations."""
        self._particles.update(dt)
        self._spheres.update(dt)
        self._goals.update(dt)
        self._field_obs.update(dt, state)
        self._mallet_fx.update(dt, players, manager.stacks)

    # --- React to manager events ------------------------------------------

    def on_manager_event(self, event: Dict) -> None:
        """
        Call this for each dict returned by PowerUpManager.update().

        Handles:
          "collected" → particle burst at collection position
          "expired"   → small white puff at last known mallet position
        """
        etype = event.get("type")
        if etype == "collected":
            pos   = event.get("position", [0, 0])
            color = event.get("particle_color", (255, 255, 255))
            self._particles.emit(pos[0], pos[1], PARTICLE_COUNT, color)
        elif etype == "expired":
            # Small white dissipation burst — position not stored; skip silently
            pass

    # --- Draw calls (order matters) --------------------------------------

    def draw_spheres(
        self,
        screen: pygame.Surface,
        manager: Any,         # PowerUpManager
    ) -> None:
        """Draw all FieldSpheres currently on the field."""
        self._spheres.draw(screen, manager.get_field_spheres())

    def draw_active_effects(
        self,
        screen: pygame.Surface,
        manager: Any,         # PowerUpManager
        players: list,
        state: Any,
        config: Any,          # GameConfig
    ) -> None:
        """Draw all per-type continuous visual effects."""
        stacks = manager.stacks
        self._goals.draw(screen, stacks, config)
        self._field_obs.draw(screen, state)
        self._mallet_fx.draw(screen, players, stacks)

    def draw_particles(self, screen: pygame.Surface) -> None:
        """Draw particle bursts (always on top of everything else)."""
        self._particles.draw(screen)

    # --- Invisibility alpha (for use by the main mallet renderer) --------

    def get_mallet_alpha(
        self,
        player_idx: int,
        stacks: list,
        is_own_screen: bool,
    ) -> int:
        """
        Return the alpha the main renderer should apply to a mallet sprite.

        Call once per mallet per frame from the game renderer:

            alpha = pu_renderer.get_mallet_alpha(i, manager.stacks, is_pvp=False)
            mallet_surf.set_alpha(alpha)
        """
        return self._mallet_fx.get_invisibility_alpha(player_idx, stacks, is_own_screen)
