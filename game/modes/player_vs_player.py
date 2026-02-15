"""Player vs Player local mode launcher."""
from game.core.match_manager import MatchConfig, GameMode


def create_config(score_limit: int = 7, time_limit: float = None,
                  powerups: bool = True) -> MatchConfig:
    """Create PvP local match config."""
    return MatchConfig(
        mode=GameMode.PLAYER_VS_PLAYER,
        score_limit=score_limit,
        time_limit_seconds=time_limit,
        powerups_enabled=powerups,
        level_id=1,
    )
