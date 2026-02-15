"""
Game mode configuration and selection.
Provides factory methods for creating MatchConfig for different modes.
"""
from game.core.match_manager import MatchConfig, GameMode


def create_pvai_config(level_id: int = 1, score_limit: int = 7,
                       time_limit: float = None, powerups: bool = False) -> MatchConfig:
    """Create a Player vs AI match configuration."""
    return MatchConfig(
        mode=GameMode.PLAYER_VS_AI,
        score_limit=score_limit,
        time_limit_seconds=time_limit,
        powerups_enabled=powerups,
        level_id=level_id,
    )


def create_pvp_config(score_limit: int = 7, time_limit: float = None,
                      powerups: bool = True) -> MatchConfig:
    """Create a Player vs Player local match configuration."""
    return MatchConfig(
        mode=GameMode.PLAYER_VS_PLAYER,
        score_limit=score_limit,
        time_limit_seconds=time_limit,
        powerups_enabled=powerups,
        level_id=1,  # PvP uses default theme
    )


def create_timed_match(mode: GameMode = GameMode.PLAYER_VS_AI,
                       time_limit: float = 120.0, level_id: int = 1,
                       powerups: bool = True) -> MatchConfig:
    """Create a time-limited match (default 2 minutes)."""
    return MatchConfig(
        mode=mode,
        score_limit=999,  # Effectively no score limit
        time_limit_seconds=time_limit,
        overtime_on_tie=True,
        powerups_enabled=powerups,
        level_id=level_id,
    )
