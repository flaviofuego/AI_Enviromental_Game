"""Player vs AI mode launcher."""
from game.core.match_manager import MatchConfig, GameMode


def create_config(level_id: int = 1, score_limit: int = 7,
                  time_limit: float = None, powerups: bool = False) -> MatchConfig:
    """Create Player vs AI match config."""
    return MatchConfig(
        mode=GameMode.PLAYER_VS_AI,
        score_limit=score_limit,
        time_limit_seconds=time_limit,
        powerups_enabled=powerups,
        level_id=level_id,
    )
