"""
Entry point: launches the game hub (main menu → level select → game).

Usage:
    python -m game.main_hub
    # or via Makefile:  make play
"""
import os
import sys

# Ensure project root is on sys.path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Re-export start_game so existing code (Level_Select.py) can do:
#   from game.main_hub import start_game
from game.main import main_with_config as start_game  # noqa: F401

# Re-export transitions for backward compat
from game.hub.transitions import ice_melt_transition, transition_effect  # noqa: F401


def main() -> None:
    """Launch the screen controller loop."""
    from game.hub.screen_controller import ScreenController

    controller = ScreenController()
    controller.run()


if __name__ == "__main__":
    main()