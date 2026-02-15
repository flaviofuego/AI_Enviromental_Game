"""
Screen controller — routes between Home, Level Select, and Game screens.
Manages the top-level navigation loop replacing the inline __main__ block.
"""
import sys

import pygame

from game.pages.home import HockeyMainScreen
from game.pages.Level_Select import LevelSelectScreen
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

        audio_manager.cleanup()
        pygame.quit()
        sys.exit()

    # ------------------------------------------------------------------
    # Pantallas
    # ------------------------------------------------------------------

    def _handle_home(self) -> None:
        audio_manager.play_music("home")

        main_screen = HockeyMainScreen(self.screen, self.save_system)
        transition_effect(self.screen, fade_out=False)

        result = main_screen.run()

        if result == "exit":
            self._running = False
        elif result == "level_select":
            audio_manager.play_sound_effect("transition")
            transition_effect(self.screen, fade_out=True)
            self._current = "level_select"

    def _handle_level_select(self) -> None:
        audio_manager.play_music("level_select")
        audio_manager.preload_audio_for_screen("level_select")

        level_screen = LevelSelectScreen(self.save_system, self.screen)
        transition_effect(self.screen, fade_out=False)

        result = level_screen.run()

        if result == "exit":
            self._running = False
        elif result == "back_to_menu":
            audio_manager.play_sound_effect("transition")
            transition_effect(self.screen, fade_out=True)
            self._current = "home"
