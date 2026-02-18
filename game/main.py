"""
Main entry point for the Air Hockey game.
Replaces main_improved.py with the new modular architecture.
"""
import os
import sys
import pygame

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from shared.config import GameConfig
from shared.powerups import PowerUpManager
from shared.powerups.registry import PowerUpRegistry
from shared.powerups.definitions import register_all
from game.core.game_engine import GameEngine
from game.core.match_manager import MatchConfig, GameMode
from game.ui.hud import HUD


def run_game(screen: pygame.Surface, match_config: MatchConfig = None,
             save_system=None, powerup_phases: list = None) -> str:
    """
    Run a single game match with the given configuration.
    Returns: 'exit', 'back_to_menu', 'retry', 'next_level'.
    """
    config = GameConfig(width=screen.get_width(), height=screen.get_height())
    if match_config is None:
        match_config = MatchConfig()

    engine = GameEngine(screen, match_config, config, save_system)

    # Attach HUD
    engine.hud = HUD(config)

    # Attach power-up manager if enabled
    if match_config.powerups_enabled:
        registry = PowerUpRegistry()
        register_all(registry)
        engine.powerup_manager = PowerUpManager(
            config, registry,
            enabled_phases=powerup_phases if powerup_phases else None,
        )

    return engine.run()


def main_with_config(screen=None, level_id=1, save_system=None, **kwargs):
    """
    Backward-compatible entry point.
    Called from game/main_hub.py and game/pages/Level_Select.py.
    Accepts level_config dict in kwargs for backward compat.
    """
    if screen is None:
        pygame.init()
        info = pygame.display.Info()
        w = min(1200, info.current_w - 100)
        h = min(800, info.current_h - 100)
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Hockey Is Melting Down")

    # Support level_config={'level_id': N} from Level_Select.py
    level_config = kwargs.get('level_config')
    if level_config and isinstance(level_config, dict):
        level_id = level_config.get('level_id', level_id)

    # Resolve powerup phases from level config
    from game.config.level_config import get_level_config
    lvl_cfg = get_level_config(level_id)
    powerup_phases = lvl_cfg.get('powerup_phases', None)
    powerups_enabled = bool(powerup_phases)

    match_config = MatchConfig(
        mode=GameMode.PLAYER_VS_AI,
        score_limit=7,
        level_id=level_id,
        powerups_enabled=powerups_enabled,
    )

    return run_game(screen, match_config, save_system, powerup_phases=powerup_phases)


def main() -> None:
    """CLI entry point for quick play (level 1, Player vs AI)."""
    pygame.init()
    info = pygame.display.Info()
    w = min(1200, info.current_w - 100)
    h = min(800, info.current_h - 100)
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Hockey Is Melting Down - Salva la Tierra")

    run_game(screen)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
