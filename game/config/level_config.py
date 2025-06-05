"""
Configuration file for game levels, defining themes and assets for each level
"""

LEVELS = {
    1: {
        "name": "Arctic Meltdown",
        "description": "El hielo ártico se derrite rápidamente. ¡Juega para salvarlo!",
        "theme": {
            "background": "background.png",
            "mallet_ai": "mallet_IA.png",
            "mallet_player": "mallet_IA.png",  # Usará el mismo pero con diferente tinte
            "puck": "puck.png",
            "goal_left": "porteria_izq.png",
            "goal_right": "porteria_der.png",
            "table_color": (173, 216, 230),  # Ice blue
            "glow_colors": {
                "player": (0, 191, 255),  # Deep sky blue
                "ai": (135, 206, 250),    # Light sky blue
                "puck": (240, 248, 255)   # White blue
            }
        },
        "difficulty": 1,
        "ai_reaction_speed": 0.25 - 0.15,  # IA máxima velocidad
        "ai_prediction_factor": 0.65 - 0.15,  # Predicción experta
        "ai_move_speed": 20 - 10
    },
    2: {
        "name": "Forest Guardian",
        "description": "Los bosques están en peligro. ¡Defiéndelos!",
        "theme": {
            "background": "background.png",
            "mallet_ai": "mallet_IA.png",
            "mallet_player": "mallet_IA.png",
            "puck": "puck.png",
            "goal_left": "porteria_izq.png",
            "goal_right": "porteria_der.png",
            "table_color": (34, 139, 34),  # Forest green
            "glow_colors": {
                "player": (50, 205, 50),   # Lime green
                "ai": (144, 238, 144),     # Light green
                "puck": (0, 100, 0)        # Dark green
            }
        },
        "difficulty": 2,
        "ai_reaction_speed": 0.25 - 0.07,  # IA máxima velocidad
        "ai_prediction_factor": 0.65 - 0.07,  # Predicción experta
        "ai_move_speed": 20 - 5
    },
    3: {
        "name": "Ocean Defense",
        "description": "Los océanos sufren contaminación. ¡Protégelos!",
        "theme": {
            "background": "background.png",
            "mallet_ai": "mallet_IA.png",
            "mallet_player": "mallet_IA.png",
            "puck": "puck.png",
            "goal_left": "porteria_izq.png",
            "goal_right": "porteria_der.png",
            "table_color": (0, 105, 148),  # Deep blue
            "glow_colors": {
                "player": (30, 144, 255),  # Dodger blue
                "ai": (0, 191, 255),       # Deep sky blue
                "puck": (135, 206, 250)    # Light sky blue
            }
        },
        "difficulty": 3,
        "ai_reaction_speed": 0.25,  # IA máxima velocidad
        "ai_prediction_factor": 0.65,  # Predicción experta
        "ai_move_speed": 20
    },
    4: {
        "name": "City Heat Island",
        "description": "Las ciudades se sobrecalientan. ¡Enfríalas!",
        "theme": {
            "background": "background.png",
            "mallet_ai": "mallet_IA.png",
            "mallet_player": "mallet_IA.png",
            "puck": "puck.png",
            "goal_left": "porteria_izq.png",
            "goal_right": "porteria_der.png",
            "table_color": (70, 70, 70),  # Urban grey
            "glow_colors": {
                "player": (255, 140, 0),   # Orange
                "ai": (220, 50, 50),       # Red
                "puck": (255, 215, 0)      # Gold
            }
        },
        "difficulty": 4,
       "ai_reaction_speed": 0.25 + 0.15,  # IA máxima velocidad
        "ai_prediction_factor": 0.65 + 0.15,  # Predicción experta
        "ai_move_speed": 20 + 5
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
            "table_color": (20, 20, 20),  # Dark
            "glow_colors": {
                "player": (147, 0, 211),   # Purple
                "ai": (255, 0, 0),         # Pure red
                "puck": (255, 255, 255)    # White
            }
        },
        "difficulty": 5,
        "ai_reaction_speed": 0.25 + 0.30,  # IA máxima velocidad
        "ai_prediction_factor": 0.65 + 0.30,  # Predicción experta
        "ai_move_speed": 20 + 10
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