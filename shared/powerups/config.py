"""
Powerup configuration constants — single source of truth for tuning.

Game designers adjust numbers here without touching any logic file.
Level configs in game/config/level_config.py can override per-level values
using the 'powerup_overrides' dict key (see docs/powerups.md §12.2).
"""

# ---------------------------------------------------------------------------
# Per-powerup base config
# ---------------------------------------------------------------------------

POWERUP_CONFIGS: dict = {
    "speed_boost": {
        "duration":    5.0,
        "multiplier":  1.3,   # +30 % speed
        "phase":       1,
    },
    "shield": {
        "duration":    5.0,
        "phase":       2,
    },
    "magnet": {
        "duration":    5.0,
        "extra_radius": 8,    # px beyond mallet radius
        "force":       0.3,   # px/frame attraction
        "phase":       3,
    },
    "duplication": {
        "duration":    7.0,
        "stack_mult":  2.0,   # doubles all active numeric multipliers
        "phase":       4,
    },
    "slow_opponent": {
        "duration":    7.0,
        "multiplier":  0.75,  # -25 % speed
        "min_cap":     0.40,  # floor enforced by EffectStack
        "phase":       5,
    },
    "obstacle": {
        "duration":    7.0,
        "count":       2,     # obstacles spawned per collection
        "radius":      20,    # px
        "phase":       6,
    },
    "paralyze": {
        "duration":    3.0,
        "phase":       7,
    },
    "invisibility": {
        "duration":    3.0,
        "alpha":       20,    # sprite alpha visible to opponent
        "phase":       8,
    },
}


# ---------------------------------------------------------------------------
# Field sphere behaviour
# ---------------------------------------------------------------------------

SPHERE_FIELD_LIFETIME    = 5.0    # seconds before auto-expiry if uncollected
SPHERE_SPEED             = 1.5    # px/frame base movement
SPHERE_DIRECTION_CHANGE  = 2.0    # seconds between random direction changes
MAX_POWERUPS_ON_FIELD    = 3      # simultaneous cap
SPAWN_INTERVAL_MIN       = 8.0    # seconds
SPAWN_INTERVAL_MAX       = 15.0   # seconds


# ---------------------------------------------------------------------------
# Particle burst on collection
# ---------------------------------------------------------------------------

PARTICLE_COUNT           = 15
PARTICLE_SPEED_MIN       = 2.0    # px/frame
PARTICLE_SPEED_MAX       = 5.0    # px/frame
PARTICLE_LIFETIME        = 0.4    # seconds


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

HUD_MAX_VISIBLE_PILLS    = 4      # beyond this, collapse to "+N"
HUD_WARN_THRESHOLD       = 1.0    # seconds remaining when pill flashes red
