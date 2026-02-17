"""
Screen controller — routes between Home, Level Select, PvP Setup, and Game screens.
Manages the top-level navigation loop replacing the inline __main__ block.
"""
import sys

import pygame

from game.pages.home import HockeyMainScreen
from game.pages.Level_Select import LevelSelectScreen
from game.pages.pvp_setup import PvPSetupScreen
from game.config.save_system import GameSaveSystem
from game.components.AudioManager import audio_manager
from game.hub.transitions import transition_effect


class ScreenController:
    """Central hub that owns the pygame window and navigates between screens."""

    def __init__(self) -> None:
        pygame.init()

        info = pygame.display.Info()
        self.width = min(1200, info.current_w - 100)
        self.height = min(800, info.current_h - 100)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Hockey Is Melting Down - Salva la Tierra")

        self.save_system = GameSaveSystem()
        self._current = "home"
        self._running = True

        # Screen instance cache — avoids full reconstruction each time
        self._screen_cache: dict[str, object] = {}
        self._screen_dirty: set[str] = set()  # screens that need rebuild

        # Cargar último perfil
        last_profile = self.save_system.get_last_used_profile()
        if last_profile:
            print(f"Cargando último perfil: {last_profile['player_name']}")

        # Música inicial
        audio_manager.preload_audio_for_screen("home")
        audio_manager.play_music("home")

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Ejecuta el loop de navegación entre pantallas hasta salir."""
        while self._running:
            if self._current == "home":
                self._handle_home()
            elif self._current == "level_select":
                self._handle_level_select()
            elif self._current == "pvp_setup":
                self._handle_pvp_setup()

        audio_manager.cleanup()
        pygame.quit()
        sys.exit()

    # ------------------------------------------------------------------
    # Pantallas
    # ------------------------------------------------------------------

    def _handle_home(self) -> None:
        audio_manager.play_music("home")

        if "home" not in self._screen_cache or "home" in self._screen_dirty:
            self._screen_cache["home"] = HockeyMainScreen(self.screen, self.save_system)
            self._screen_dirty.discard("home")
        main_screen = self._screen_cache["home"]
        transition_effect(self.screen, fade_out=False)

        result = main_screen.run()

        if result == "exit":
            self._running = False
        elif result == "level_select":
            audio_manager.play_sound_effect("transition")
            transition_effect(self.screen, fade_out=True)
            self._current = "level_select"
        elif result == "pvp_setup":
            audio_manager.play_sound_effect("transition")
            transition_effect(self.screen, fade_out=True)
            self._current = "pvp_setup"

    def _handle_level_select(self) -> None:
        audio_manager.play_music("level_select")
        audio_manager.preload_audio_for_screen("level_select")

        if "level_select" not in self._screen_cache or "level_select" in self._screen_dirty:
            self._screen_cache["level_select"] = LevelSelectScreen(self.save_system, self.screen)
            self._screen_dirty.discard("level_select")
        level_screen = self._screen_cache["level_select"]
        # Re-sync profile data in case it changed
        level_screen.current_profile = self.save_system.current_profile
        level_screen.load_levels_status()
        transition_effect(self.screen, fade_out=False)

        result = level_screen.run()

        if result == "exit":
            self._running = False
        elif result == "back_to_menu":
            audio_manager.play_sound_effect("transition")
            transition_effect(self.screen, fade_out=True)
            self._current = "home"

    def _handle_pvp_setup(self) -> None:
        """Handle PvP setup screen. On 'play', launches a local 2-player match."""
        audio_manager.play_music("home")

        pvp_screen = PvPSetupScreen(self.screen, self.save_system)
        transition_effect(self.screen, fade_out=False)

        result = pvp_screen.run()

        if result == "exit":
            self._running = False
        elif result == "back":
            audio_manager.play_sound_effect("transition")
            transition_effect(self.screen, fade_out=True)
            self._current = "home"
        elif isinstance(result, dict):
            # result is the match config dict from PvPSetupScreen
            # Launch the PvP game
            audio_manager.play_sound_effect("transition")
            transition_effect(self.screen, fade_out=True)
            self._launch_pvp_game(result)

    def _launch_pvp_game(self, config: dict) -> None:
        """Launch a local PvP match with the given configuration."""
        try:
            from game.main import run_game
            from game.core.match_manager import MatchConfig, GameMode

            level_id = config.get("background_level_id", 1)

            match_config = MatchConfig(
                mode=GameMode.PLAYER_VS_PLAYER,
                score_limit=config.get("score_limit", 7),
                time_limit_seconds=config.get("time_limit"),
                powerups_enabled=config.get("powerups", True),
                level_id=level_id,
            )

            run_game(self.screen, match_config, self.save_system)
        except Exception as e:
            print(f"Error al iniciar partida PvP: {e}")

        # Invalidate screens that may need refreshing after a game
        self._screen_dirty.update({"home", "level_select"})
        # Return to home after the game
        self._current = "home"
