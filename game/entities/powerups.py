"""
Power-up and limiter system with climate change environmental themes.
Power-ups spawn on the field and are collected by mallets or puck contact.
"""
import random
import math
import pygame
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional

from shared.config import GameConfig, COLORS


class PowerUpType(Enum):
    """Types of power-ups with environmental themes."""
    SPEED_BOOST = auto()       # Clean wind current
    SIZE_INCREASE = auto()     # Tree growth
    SLOW_OPPONENT = auto()     # Pollution
    PUCK_MAGNET = auto()       # Earth's magnetic field
    SHIELD = auto()            # Coral barrier
    SHRINK_OPPONENT = auto()   # Drought


@dataclass
class PowerUpEffect:
    """Active effect on a player."""
    type: PowerUpType
    remaining: float  # seconds remaining
    target_index: int  # 0 = player who collected, 1 = opponent


# Configuration per power-up type
POWERUP_CONFIG = {
    PowerUpType.SPEED_BOOST: {
        "name": "Viento Limpio",
        "description": "+50% velocidad",
        "duration": 5.0,
        "color": (100, 220, 255),
        "icon_char": "W",
        "affects_self": True,
        "multiplier": 1.5,
    },
    PowerUpType.SIZE_INCREASE: {
        "name": "Crecimiento",
        "description": "+30% tamano",
        "duration": 7.0,
        "color": (80, 200, 80),
        "icon_char": "T",
        "affects_self": True,
        "multiplier": 1.3,
    },
    PowerUpType.SLOW_OPPONENT: {
        "name": "Contaminacion",
        "description": "-30% velocidad rival",
        "duration": 5.0,
        "color": (150, 100, 50),
        "icon_char": "P",
        "affects_self": False,
        "multiplier": 0.7,
    },
    PowerUpType.PUCK_MAGNET: {
        "name": "Campo Magnetico",
        "description": "Atrae el puck",
        "duration": 4.0,
        "color": (200, 50, 200),
        "icon_char": "M",
        "affects_self": True,
        "multiplier": 1.0,
    },
    PowerUpType.SHIELD: {
        "name": "Barrera de Coral",
        "description": "Protege la porteria",
        "duration": 3.0,
        "color": (255, 150, 100),
        "icon_char": "S",
        "affects_self": True,
        "multiplier": 1.0,
    },
    PowerUpType.SHRINK_OPPONENT: {
        "name": "Sequia",
        "description": "-25% tamano rival",
        "duration": 5.0,
        "color": (220, 180, 50),
        "icon_char": "D",
        "affects_self": False,
        "multiplier": 0.75,
    },
}


class PowerUp(pygame.sprite.Sprite):
    """A single power-up item on the field."""

    def __init__(self, ptype: PowerUpType, x: float, y: float, config: GameConfig):
        super().__init__()
        self.type = ptype
        self.config = config
        self.position = [x, y]
        self.radius = int(18 * config.scale_factor)
        self.lifetime = 15.0  # seconds before disappearing if uncollected
        self.age = 0.0
        self.cfg = POWERUP_CONFIG[ptype]

        # Create sprite
        size = self.radius * 2
        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        self._draw_icon()
        self.rect = self.image.get_rect(center=(int(x), int(y)))
        self.mask = pygame.mask.from_surface(self.image)

        # Animation
        self._pulse_phase = random.uniform(0, math.pi * 2)

    def _draw_icon(self):
        """Draw the power-up icon."""
        color = self.cfg["color"]
        r = self.radius
        # Outer glow
        pygame.draw.circle(self.image, (*color, 60), (r, r), r)
        # Inner circle
        pygame.draw.circle(self.image, (*color, 200), (r, r), int(r * 0.7))
        # Border
        pygame.draw.circle(self.image, COLORS.WHITE, (r, r), int(r * 0.7), 2)
        # Icon letter
        font = pygame.font.Font(None, int(r * 1.2))
        txt = font.render(self.cfg["icon_char"], True, COLORS.WHITE)
        tx = r - txt.get_width() // 2
        ty = r - txt.get_height() // 2
        self.image.blit(txt, (tx, ty))

    def update_animation(self, dt: float):
        """Animate the power-up (pulsing glow)."""
        self.age += dt
        self._pulse_phase += dt * 3
        # Regenerate sprite with pulse effect
        size = self.radius * 2
        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        pulse = 0.8 + 0.2 * math.sin(self._pulse_phase)
        color = self.cfg["color"]
        r = self.radius
        alpha = int(60 * pulse)
        pygame.draw.circle(self.image, (*color, alpha), (r, r), r)
        inner_r = int(r * 0.7 * pulse)
        pygame.draw.circle(self.image, (*color, 200), (r, r), max(1, inner_r))
        pygame.draw.circle(self.image, COLORS.WHITE, (r, r), max(1, inner_r), 2)
        font = pygame.font.Font(None, int(r * 1.2))
        txt = font.render(self.cfg["icon_char"], True, COLORS.WHITE)
        self.image.blit(txt, (r - txt.get_width() // 2, r - txt.get_height() // 2))

    def is_expired(self) -> bool:
        return self.age >= self.lifetime

    def check_collision(self, entity) -> bool:
        """Check if entity (mallet) collects this power-up."""
        dx = self.position[0] - entity.position[0]
        dy = self.position[1] - entity.position[1]
        dist = math.hypot(dx, dy)
        return dist <= self.radius + entity.radius


class PowerUpManager:
    """Manages spawning, updating, and applying power-ups."""

    def __init__(self, config: GameConfig):
        self.config = config
        self.field_powerups: List[PowerUp] = []
        self.active_effects: List[PowerUpEffect] = []
        self.spawn_timer = 0.0
        self.spawn_interval_min = 10.0
        self.spawn_interval_max = 20.0
        self.next_spawn_time = random.uniform(self.spawn_interval_min, self.spawn_interval_max)
        self.max_on_field = 2
        # Shield state
        self.shield_active = [False, False]  # [player1, player2]

    def update(self, dt: float, players: list, puck):
        """Update all power-ups and effects."""
        self.spawn_timer += dt

        # Spawn new power-ups
        if self.spawn_timer >= self.next_spawn_time and len(self.field_powerups) < self.max_on_field:
            self._spawn_random()
            self.spawn_timer = 0.0
            self.next_spawn_time = random.uniform(self.spawn_interval_min, self.spawn_interval_max)

        # Update animations and check expiry
        expired = []
        for pu in self.field_powerups:
            pu.update_animation(dt)
            if pu.is_expired():
                expired.append(pu)

        for pu in expired:
            self.field_powerups.remove(pu)

        # Check collection by mallets
        for i, player in enumerate(players):
            collected = []
            for pu in self.field_powerups:
                if pu.check_collision(player):
                    self._apply_powerup(pu, i, players, puck)
                    collected.append(pu)
            for pu in collected:
                self.field_powerups.remove(pu)

        # Update active effects
        expired_effects = []
        for effect in self.active_effects:
            effect.remaining -= dt
            if effect.remaining <= 0:
                expired_effects.append(effect)

        for effect in expired_effects:
            self._remove_effect(effect, players)
            self.active_effects.remove(effect)

        # Apply magnet effect
        for effect in self.active_effects:
            if effect.type == PowerUpType.PUCK_MAGNET:
                player = players[effect.target_index]
                dx = player.position[0] - puck.position[0]
                dy = player.position[1] - puck.position[1]
                dist = math.hypot(dx, dy)
                if dist > 10:
                    force = 0.3
                    puck.velocity[0] += (dx / dist) * force
                    puck.velocity[1] += (dy / dist) * force

    def _spawn_random(self):
        """Spawn a random power-up in a safe area of the field."""
        W, H = self.config.width, self.config.height
        # Spawn in the middle third of the field, avoiding edges and goals
        x = random.uniform(W * 0.25, W * 0.75)
        y = random.uniform(H * 0.15, H * 0.85)
        ptype = random.choice(list(PowerUpType))
        pu = PowerUp(ptype, x, y, self.config)
        self.field_powerups.append(pu)

    def _apply_powerup(self, pu: PowerUp, collector_idx: int, players: list, puck):
        """Apply power-up effect when collected."""
        cfg = POWERUP_CONFIG[pu.type]
        target_idx = collector_idx if cfg["affects_self"] else (1 - collector_idx)

        effect = PowerUpEffect(
            type=pu.type,
            remaining=cfg["duration"],
            target_index=target_idx if cfg["affects_self"] else (1 - collector_idx),
        )

        if pu.type == PowerUpType.SPEED_BOOST:
            players[collector_idx].speed_multiplier = cfg["multiplier"]
            effect.target_index = collector_idx
        elif pu.type == PowerUpType.SIZE_INCREASE:
            players[collector_idx].apply_size_modifier(cfg["multiplier"])
            effect.target_index = collector_idx
        elif pu.type == PowerUpType.SLOW_OPPONENT:
            opponent = 1 - collector_idx
            players[opponent].speed_multiplier = cfg["multiplier"]
            effect.target_index = opponent
        elif pu.type == PowerUpType.SHRINK_OPPONENT:
            opponent = 1 - collector_idx
            players[opponent].apply_size_modifier(cfg["multiplier"])
            effect.target_index = opponent
        elif pu.type == PowerUpType.SHIELD:
            self.shield_active[collector_idx] = True
            effect.target_index = collector_idx
        elif pu.type == PowerUpType.PUCK_MAGNET:
            effect.target_index = collector_idx

        self.active_effects.append(effect)

    def _remove_effect(self, effect: PowerUpEffect, players: list):
        """Remove an expired effect."""
        if effect.type == PowerUpType.SPEED_BOOST:
            players[effect.target_index].speed_multiplier = 1.0
        elif effect.type == PowerUpType.SIZE_INCREASE:
            players[effect.target_index].apply_size_modifier(1.0)
        elif effect.type == PowerUpType.SLOW_OPPONENT:
            players[effect.target_index].speed_multiplier = 1.0
        elif effect.type == PowerUpType.SHRINK_OPPONENT:
            players[effect.target_index].apply_size_modifier(1.0)
        elif effect.type == PowerUpType.SHIELD:
            self.shield_active[effect.target_index] = False

    def draw(self, screen: pygame.Surface):
        """Draw all field power-ups."""
        for pu in self.field_powerups:
            screen.blit(pu.image, pu.rect)

    def get_active_effects_for_player(self, player_idx: int) -> List[PowerUpEffect]:
        """Get all active effects for a specific player."""
        return [e for e in self.active_effects if e.target_index == player_idx]

    def reset(self):
        """Clear all power-ups and effects."""
        self.field_powerups.clear()
        self.active_effects.clear()
        self.shield_active = [False, False]
        self.spawn_timer = 0.0
