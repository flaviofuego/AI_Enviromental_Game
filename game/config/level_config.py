"""
Configuration file for game levels, defining themes and assets for each level
"""

LEVELS = {
    1: {
        "name": "Arctic Meltdown",
        "description": "El hielo ártico se derrite rápidamente. ¡Juega para salvarlo!",
        "theme": {
            "background": "arctic_bg.png",
            "mallet": "ice_mallet.png",
            "puck": "snowflake_puck.png",
            "table_color": (173, 216, 230),  # Ice blue
            "glow_colors": {
                "player": (0, 191, 255),  # Deep sky blue
                "ai": (135, 206, 250),    # Light sky blue
                "puck": (240, 248, 255)   # White blue
            }
        },
        "difficulty": 1
    },
    2: {
        "name": "Forest Guardian",
        "description": "Los bosques están en peligro. ¡Defiéndelos!",
        "theme": {
            "background": "forest_bg.png",
            "mallet": "leaf_mallet.png",
            "puck": "seed_puck.png",
            "table_color": (34, 139, 34),  # Forest green
            "glow_colors": {
                "player": (50, 205, 50),   # Lime green
                "ai": (144, 238, 144),     # Light green
                "puck": (0, 100, 0)        # Dark green
            }
        },
        "difficulty": 2
    },
    3: {
        "name": "Ocean Defense",
        "description": "Los océanos sufren contaminación. ¡Protégelos!",
        "theme": {
            "background": "ocean_bg.png",
            "mallet": "wave_mallet.png",
            "puck": "bubble_puck.png",
            "table_color": (0, 105, 148),  # Deep blue
            "glow_colors": {
                "player": (30, 144, 255),  # Dodger blue
                "ai": (0, 191, 255),       # Deep sky blue
                "puck": (135, 206, 250)    # Light sky blue
            }
        },
        "difficulty": 3
    }
}

def get_level_config(level_id):
    """Get the configuration for a specific level"""
    return LEVELS.get(level_id, LEVELS[1])  # Default to level 1 if not found

def get_asset_path(level_id, asset_name):
    """Get the full path for a level-specific asset"""
    import os
    
    # Base assets directory
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "levels", str(level_id))
    
    # Create directory if it doesn't exist
    os.makedirs(assets_dir, exist_ok=True)
    
    return os.path.join(assets_dir, asset_name) 