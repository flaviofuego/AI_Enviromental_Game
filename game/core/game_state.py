"""
Game state machine: manages transitions between menu, playing, paused, and game_over.
"""
from enum import Enum, auto

from game.core.match_timer import MatchTimer


class GamePhase(Enum):
    """Top-level game phases."""
    MENU = auto()
    PLAYING = auto()
    PAUSED = auto()
    GAME_OVER = auto()
    COUNTDOWN = auto()  # Brief countdown before resuming


class GameState:
    """Tracks all runtime game state."""

    def __init__(self):
        self.phase = GamePhase.MENU
        self.player_score = 0
        self.ai_score = 0
        self.winner = None
        self.debug_mode = False
        self.show_fps = False
        # Match stats
        self.total_hits_player = 0
        self.total_hits_ai = 0
        # Centralised timer — replaces raw match_start_time / match_elapsed
        self.timer = MatchTimer()

    # ------------------------------------------------------------------
    # Backward-compatible properties so existing code keeps working
    # ------------------------------------------------------------------

    @property
    def match_start_time(self) -> float:
        """Deprecated — use ``self.timer.elapsed`` instead."""
        return self.timer._start_time

    @match_start_time.setter
    def match_start_time(self, value: float) -> None:
        # Kept for legacy callers; prefer timer.start()
        self.timer._start_time = value

    @property
    def match_elapsed(self) -> float:
        return self.timer.elapsed

    @match_elapsed.setter
    def match_elapsed(self, value: float) -> None:
        # Legacy setter — freeze the timer at this value
        self.timer._frozen_elapsed = value

    def reset_match(self):
        """Reset scores and stats for a new match."""
        self.player_score = 0
        self.ai_score = 0
        self.winner = None
        self.total_hits_player = 0
        self.total_hits_ai = 0
        self.timer.start()
        self.phase = GamePhase.PLAYING

    def record_goal(self, scorer: str, score_limit: int):
        """
        Record a goal. Returns True if the match is over.
        scorer: 'player' or 'ai'
        """
        if scorer == "player":
            self.player_score += 1
        elif scorer == "ai":
            self.ai_score += 1

        if self.player_score >= score_limit:
            self.winner = "player"
            self.phase = GamePhase.GAME_OVER
            self.timer.freeze()
            return True
        if self.ai_score >= score_limit:
            self.winner = "ai"
            self.phase = GamePhase.GAME_OVER
            self.timer.freeze()
            return True
        return False

    def check_time_limit(self, elapsed: float, time_limit: float, overtime_on_tie: bool = True) -> bool:
        """
        Check if time limit has been reached. Returns True if match should end.
        """
        if time_limit is None or time_limit <= 0:
            return False
        if elapsed >= time_limit:
            if self.player_score != self.ai_score or not overtime_on_tie:
                self.winner = "player" if self.player_score > self.ai_score else "ai"
                if self.player_score == self.ai_score:
                    self.winner = "tie"
                self.phase = GamePhase.GAME_OVER
                self.timer.freeze()
                return True
        return False
