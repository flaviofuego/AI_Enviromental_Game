"""
Configuration file for game levels, defining themes, mechanics and assets for each level.

Level Mechanics Design:
─────────────────────────────────────────────────────────────────────────
Level 1 — Arctic Meltdown (tutorial):
  Base air hockey. No special mechanics. Slow AI.
  Environmental message: Plastic in oceans.

Level 2 — Ozone Shield:
  MECHANIC: UV Zones — Periodic UV radiation zones appear on the field.
  Puck speeds up 30% when passing through a UV zone. Zones pulse and move.
  AI is moderately faster. Player must learn positioning.
  Environmental message: CFC gases and ozone layer.

Level 3 — Smog Storm:
  MECHANIC: Fog of War — Visibility is reduced. A fog layer covers most
  of the field; only a radius around the player's mallet and around the
  puck is clear. The AI "sees" through the fog (advantage).
  Environmental message: Air pollution in cities.

Level 4 — Vanishing Forest:
  MECHANIC: Shrinking Field — The playable field area slowly shrinks over
  time (walls close in). Every player goal "plants trees" and pushes walls
  back slightly. AI goals accelerate the shrinking.
  Environmental message: Deforestation.

Level 5 — Urban Heat Island (Final Stand):
  MECHANIC: Heat Waves — The field periodically distorts with heat wave
  effect. During heat waves the puck's friction decreases (slides faster
  and becomes harder to control). The puck also leaves a visible heat
  trail. AI is at maximum difficulty.
  Environmental message: Urban heat islands.
─────────────────────────────────────────────────────────────────────────
"""

LEVELS = {
    1: {
        "name": "Arctic Meltdown",
        "description": "El hielo ártico se derrite rápidamente. ¡Juega para salvarlo!",
        "theme": {
            "background": "background.png",
            "mallet_ai": "mallet_IA.png",
            "mallet_player": "mallet_IA.png",
            "puck": "puck.png",
            "goal_left": "porteria_izq.png",
            "goal_right": "porteria_der.png",
            "table_color": (173, 216, 230),  # Ice blue
            "glow_colors": {
                "player": (0, 191, 255),
                "ai": (135, 206, 250),
                "puck": (240, 248, 255)
            }
        },
        "mechanics": {
            "type": "none",  # Tutorial level — pure air hockey
        },
        "difficulty": 1,
        "ai_reaction_speed": 0.10,
        "ai_prediction_factor": 0.50,
        "ai_move_speed": 10
    },
    2: {
        "name": "Ozone Shield",
        "description": "Los gases CFC han abierto un cráter en el cielo. ¡Restaura el escudo!",
        "theme": {
            "background": "background.png",
            "mallet_ai": "mallet_IA.png",
            "mallet_player": "mallet_IA.png",
            "puck": "puck.png",
            "goal_left": "porteria_izq.png",
            "goal_right": "porteria_der.png",
            "table_color": (34, 139, 34),
            "glow_colors": {
                "player": (50, 205, 50),
                "ai": (144, 238, 144),
                "puck": (0, 100, 0)
            }
        },
        "mechanics": {
            "type": "none",  # UV zones desactivado — solo powerups
            # "type": "uv_zones",
            # "zone_count": 2,           # Number of UV zones on field
            # "zone_radius": 60,         # Radius of each zone (px at 800x500)
            # "speed_boost": 1.3,        # Puck speed multiplier inside zone
            # "zone_move_speed": 0.5,    # How fast zones drift
            # "zone_pulse_rate": 2.0,    # Pulse animation speed (Hz)
        },
        "powerup_phases": [1, 2],      # speed_boost + shield
        "difficulty": 2,
        "ai_reaction_speed": 0.18,
        "ai_prediction_factor": 0.58,
        "ai_move_speed": 15
    },
    3: {
        "name": "Smog Storm",
        "description": "La niebla tóxica asfixia las ciudades. ¡Purifica el aire!",
        "theme": {
            "background": "background.png",
            "mallet_ai": "mallet_IA.png",
            "mallet_player": "mallet_IA.png",
            "puck": "puck.png",
            "goal_left": "porteria_izq.png",
            "goal_right": "porteria_der.png",
            "table_color": (0, 105, 148),
            "glow_colors": {
                "player": (30, 144, 255),
                "ai": (0, 191, 255),
                "puck": (135, 206, 250)
            }
        },
        "mechanics": {
            "type": "nonr",
            #"player_vision_radius": 120,   # Clear area around player mallet
            #"puck_vision_radius": 80,      # Clear area around puck
            #"fog_opacity": 200,            # 0-255 fog darkness
            #"ai_sees_through": True,       # AI ignores fog (difficulty)
        },
        "powerup_phases": [1, 5],      # speed_boost + slow_opponent
        "difficulty": 3,
        "ai_reaction_speed": 0.25,
        "ai_prediction_factor": 0.65,
        "ai_move_speed": 20
    },
    4: {
        "name": "Vanishing Forest",
        "description": "Los bosques desaparecen. ¡Planta árboles con cada gol!",
        "theme": {
            "background": "background.png",
            "mallet_ai": "mallet_IA.png",
            "mallet_player": "mallet_IA.png",
            "puck": "puck.png",
            "goal_left": "porteria_izq.png",
            "goal_right": "porteria_der.png",
            "table_color": (70, 70, 70),
            "glow_colors": {
                "player": (255, 140, 0),
                "ai": (220, 50, 50),
                "puck": (255, 215, 0)
            }
        },
        "mechanics": {
            "type": "none",  # shrinking_field desactivado — solo powerups
            # "type": "shrinking_field",
            # "shrink_rate": 0.3,            # Pixels per second walls close in
            # "player_goal_expand": 15,      # Pixels walls push back per player goal
            # "ai_goal_shrink": 10,          # Extra shrink per AI goal
            # "min_field_ratio": 0.6,        # Minimum field size (60% of original)
        },
        "powerup_phases": [3, 6],      # magnet + obstacle
        "difficulty": 4,
        "ai_reaction_speed": 0.40,
        "ai_prediction_factor": 0.80,
        "ai_move_speed": 25
    },
    5: {
        "name": "Final Stand",
        "description": "¡El destino del planeta está en tus manos!",
        "theme": {
            "background": "background.png",
            "mallet_ai": "mallet_IA.png",
            "mallet_player": "mallet_IA.png",
            "puck": "puck.png",
            "goal_left": "porteria_izq.png",
            "goal_right": "porteria_der.png",
            "table_color": (20, 20, 20),
            "glow_colors": {
                "player": (147, 0, 211),
                "ai": (255, 0, 0),
                "puck": (255, 255, 255)
            }
        },
        "mechanics": {
            "type": "none",  # heat_waves desactivado — solo powerups
            # "type": "heat_waves",
            # "wave_interval": 8.0,          # Seconds between heat waves
            # "wave_duration": 3.0,          # How long each wave lasts
            # "friction_reduction": 0.5,     # Multiply friction during wave (slippery)
            # "visual_distortion": True,     # Enable visual heat shimmer effect
            # "trail_enabled": True,         # Puck leaves heat trail
        },
        "powerup_phases": [7, 8],      # paralyze + invisibility
        "difficulty": 5,
        "ai_reaction_speed": 0.55,
        "ai_prediction_factor": 0.95,
        "ai_move_speed": 30
    }
}

def get_level_config(level_id):
    """Get the configuration for a specific level"""
    return LEVELS.get(level_id, LEVELS[1])  # Default to level 1 if not found

def get_asset_path(level_id, asset_name):
    """Get the full path for a level-specific asset"""
    import os
    
    # Base assets directory
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "niveles", str(level_id))
    
    # Create directory if it doesn't exist
    os.makedirs(assets_dir, exist_ok=True)
    
    return os.path.join(assets_dir, asset_name) 