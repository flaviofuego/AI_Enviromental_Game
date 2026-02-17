"""
Model comparison utilities — evaluate, compare, and rank trained models.

Supports PPO and DQN models. Provides granular metrics beyond mean_reward:
win rate, goals scored/conceded, episode length, and tabular formatting.

Usage:
    from training.analysis.model_comparison import ModelEvaluator

    evaluator = ModelEvaluator(env_factory=lambda: AirHockeyEnv())
    results = evaluator.evaluate_all(["models/v2/best_model", "models/v3/best_model"])
    evaluator.print_comparison(results)
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy


# ─────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Rich evaluation result for a single model."""
    model_path: str
    model_name: str
    algorithm: str
    mean_reward: float = 0.0
    std_reward: float = 0.0
    win_rate: float = 0.0
    avg_goals_scored: float = 0.0
    avg_goals_conceded: float = 0.0
    avg_episode_length: float = 0.0
    n_episodes: int = 0
    metadata: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def goal_diff(self) -> float:
        return self.avg_goals_scored - self.avg_goals_conceded


# ─────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────

_ALGO_CLASSES = {"PPO": PPO, "DQN": DQN}


def load_model(path: str) -> tuple:
    """Load a model (PPO or DQN) from path. Returns (model, algo_name)."""
    for name, cls in _ALGO_CLASSES.items():
        try:
            return cls.load(path), name
        except Exception:
            continue
    return None, None


def load_metadata(model_dir: str) -> dict:
    """Load metadata.json from a model directory (if it exists)."""
    meta_path = os.path.join(model_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    # Try parent directory
    parent_meta = os.path.join(os.path.dirname(model_dir), "metadata.json")
    if os.path.isfile(parent_meta):
        with open(parent_meta) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """Evaluate and compare RL models with granular metrics.

    Args:
        env_factory: Callable that returns a fresh environment.
        n_eval_episodes: Episodes per model evaluation.
    """

    def __init__(
        self,
        env_factory: Callable,
        n_eval_episodes: int = 20,
    ) -> None:
        self._env_factory = env_factory
        self._n_episodes = n_eval_episodes

    def evaluate_model(self, model_path: str) -> ModelResult:
        """Evaluate a single model and return rich metrics."""
        model_name = os.path.basename(os.path.dirname(model_path)) or os.path.basename(model_path)
        model, algo = load_model(model_path)

        if model is None:
            return ModelResult(
                model_path=model_path,
                model_name=model_name,
                algorithm="unknown",
                error="Could not load model",
            )

        env = self._env_factory()
        metadata = load_metadata(os.path.dirname(model_path))

        # Run episodes manually for granular metrics
        rewards = []
        goals_scored = []
        goals_conceded = []
        episode_lengths = []
        wins = 0

        for _ in range(self._n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_steps += 1

            rewards.append(ep_reward)
            episode_lengths.append(ep_steps)

            unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
            ai_score = getattr(unwrapped, "ai_score", 0)
            player_score = getattr(unwrapped, "player_score", 0)
            goals_scored.append(ai_score)
            goals_conceded.append(player_score)
            if ai_score > player_score:
                wins += 1

        env.close()
        n = max(1, len(rewards))

        return ModelResult(
            model_path=model_path,
            model_name=model_name,
            algorithm=algo,
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            win_rate=wins / n,
            avg_goals_scored=float(np.mean(goals_scored)),
            avg_goals_conceded=float(np.mean(goals_conceded)),
            avg_episode_length=float(np.mean(episode_lengths)),
            n_episodes=n,
            metadata=metadata,
        )

    def evaluate_all(self, model_paths: list[str]) -> list[ModelResult]:
        """Evaluate multiple models and return sorted results (best first)."""
        results = []
        for path in model_paths:
            print(f"Evaluating: {path} ...")
            result = self.evaluate_model(path)
            results.append(result)
            if result.error:
                print(f"  ERROR: {result.error}")
            else:
                print(f"  reward={result.mean_reward:.2f}±{result.std_reward:.2f} "
                      f"win_rate={result.win_rate:.0%} "
                      f"goals={result.avg_goals_scored:.1f}/{result.avg_goals_conceded:.1f}")

        # Sort by mean_reward descending
        results.sort(key=lambda r: r.mean_reward, reverse=True)
        return results

    @staticmethod
    def print_comparison(results: list[ModelResult]) -> None:
        """Print a formatted comparison table."""
        if not results:
            print("No results to compare.")
            return

        print(f"\n{'=' * 90}")
        print(f"{'Model Comparison':^90}")
        print(f"{'=' * 90}")
        header = (
            f"{'Rank':<5} {'Model':<25} {'Algo':<5} {'Reward':>10} "
            f"{'WinRate':>8} {'Goals':>8} {'GD':>5} {'EpLen':>7}"
        )
        print(header)
        print("-" * 90)

        for i, r in enumerate(results, 1):
            if r.error:
                print(f"{i:<5} {r.model_name:<25} {'ERR':<5} {r.error}")
                continue
            goals = f"{r.avg_goals_scored:.1f}/{r.avg_goals_conceded:.1f}"
            print(
                f"{i:<5} {r.model_name:<25} {r.algorithm:<5} "
                f"{r.mean_reward:>10.2f} {r.win_rate:>7.0%} "
                f"{goals:>8} {r.goal_diff:>+5.1f} {r.avg_episode_length:>7.0f}"
            )

        print(f"{'=' * 90}")
        if results and not results[0].error:
            best = results[0]
            print(f"Best: {best.model_name} ({best.algorithm}) — "
                  f"reward={best.mean_reward:.2f}, win_rate={best.win_rate:.0%}")
        print()

    @staticmethod
    def results_to_dict(results: list[ModelResult]) -> list[dict]:
        """Convert results to serializable dicts for JSON export."""
        return [
            {
                "rank": i + 1,
                "model_name": r.model_name,
                "model_path": r.model_path,
                "algorithm": r.algorithm,
                "mean_reward": round(r.mean_reward, 2),
                "std_reward": round(r.std_reward, 2),
                "win_rate": round(r.win_rate, 4),
                "avg_goals_scored": round(r.avg_goals_scored, 2),
                "avg_goals_conceded": round(r.avg_goals_conceded, 2),
                "goal_diff": round(r.goal_diff, 2),
                "avg_episode_length": round(r.avg_episode_length, 1),
                "n_episodes": r.n_episodes,
                "metadata": r.metadata,
                "error": r.error,
            }
            for i, r in enumerate(results)
        ]


# ─────────────────────────────────────────────────────────────────────
# Backward-compatible API
# ─────────────────────────────────────────────────────────────────────

def compare_models(model_paths: list, env_factory, n_eval_episodes: int = 20):
    """Evaluate multiple models (backward-compatible wrapper).

    Returns list of dicts with model_path, algo, mean_reward, std_reward.
    """
    evaluator = ModelEvaluator(env_factory, n_eval_episodes)
    results = evaluator.evaluate_all(model_paths)
    evaluator.print_comparison(results)
    return evaluator.results_to_dict(results)
