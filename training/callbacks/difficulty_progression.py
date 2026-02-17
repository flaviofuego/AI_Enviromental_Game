"""Callback for adaptive difficulty progression during training.

Uses ``CurriculumMetrics`` for fine-grained advancement decisions
and integrates with the modular ``AlgorithmicOpponent`` system.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from training.configs.curriculum import (
    CURRICULUM_LEVELS,
    CurriculumMetrics,
    DifficultyLevel,
)
from training.envs.opponents import OpponentParams


class DifficultyProgressionCallback(BaseCallback):
    """Adjusts opponent difficulty based on granular agent performance metrics.

    Evaluates the agent periodically, collects ``CurriculumMetrics``, and
    advances the curriculum level when thresholds are met.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 5,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -float("inf")
        self._current_level_idx = 0

        # Sync training env with starting curriculum level
        self._apply_level(CURRICULUM_LEVELS[0])

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        metrics = self._evaluate()

        level = CURRICULUM_LEVELS[self._current_level_idx]
        skill = level.opponent_skill

        if metrics.meets_advancement(level):
            if self._current_level_idx < len(CURRICULUM_LEVELS) - 1:
                self._current_level_idx += 1
                new_level = CURRICULUM_LEVELS[self._current_level_idx]
                self._apply_level(new_level)
                skill = new_level.opponent_skill
                if self.verbose > 0:
                    print(
                        f"[Curriculum] Step {self.n_calls}: "
                        f"Advanced to level {self._current_level_idx} "
                        f"(skill={skill:.2f}) | "
                        f"win_rate={metrics.win_rate:.2f} "
                        f"goals={metrics.avg_goals_scored:.1f}/{metrics.avg_goals_conceded:.1f}"
                    )
            else:
                if self.verbose > 0:
                    print(
                        f"[Curriculum] Step {self.n_calls}: "
                        f"Already at max level {self._current_level_idx} | "
                        f"win_rate={metrics.win_rate:.2f}"
                    )
        elif self.verbose > 0:
            print(
                f"[Curriculum] Step {self.n_calls}: "
                f"Level {self._current_level_idx} (skill={skill:.2f}) | "
                f"win_rate={metrics.win_rate:.2f} "
                f"goals={metrics.avg_goals_scored:.1f}/{metrics.avg_goals_conceded:.1f} "
                f"— not advancing"
            )

        # Also update best_mean_reward for legacy compatibility
        if metrics.mean_reward > self.best_mean_reward:
            self.best_mean_reward = metrics.mean_reward

        return True

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self) -> CurriculumMetrics:
        """Run evaluation episodes and collect granular metrics."""
        rewards: list[float] = []
        goals_scored: list[int] = []
        goals_conceded: list[int] = []
        wins = 0

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward

            rewards.append(ep_reward)

            # Extract scores from the unwrapped env
            unwrapped = self.eval_env.unwrapped if hasattr(self.eval_env, "unwrapped") else self.eval_env
            ai_score = getattr(unwrapped, "ai_score", 0)
            player_score = getattr(unwrapped, "player_score", 0)
            goals_scored.append(ai_score)
            goals_conceded.append(player_score)
            if ai_score > player_score:
                wins += 1

        n = max(1, len(rewards))
        return CurriculumMetrics(
            win_rate=wins / n,
            avg_goals_scored=float(np.mean(goals_scored)) if goals_scored else 0.0,
            avg_goals_conceded=float(np.mean(goals_conceded)) if goals_conceded else 0.0,
            mean_reward=float(np.mean(rewards)),
            episodes_evaluated=n,
        )

    # ------------------------------------------------------------------
    # Apply difficulty level
    # ------------------------------------------------------------------

    def _apply_level(self, level: DifficultyLevel) -> None:
        """Apply a curriculum level to both training and eval environments."""
        params = OpponentParams.from_curriculum_level(level)

        for env in (self.training_env, self.eval_env):
            unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
            if hasattr(unwrapped, "opponent"):
                unwrapped.opponent.params = params
                unwrapped.opponent_skill = level.opponent_skill
            elif hasattr(unwrapped, "increase_opponent_difficulty"):
                unwrapped.opponent_skill = level.opponent_skill
