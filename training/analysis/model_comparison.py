"""
Model comparison utilities — evaluate and compare trained models.
Migrated from compare_models/ directory.
"""
import os
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy


def load_model(path: str):
    """Load a model (PPO or DQN) from path."""
    try:
        return PPO.load(path), "PPO"
    except Exception:
        try:
            return DQN.load(path), "DQN"
        except Exception:
            return None, None


def compare_models(model_paths: list, env_factory, n_eval_episodes: int = 20):
    """
    Evaluate multiple models on the same environment and return comparison results.

    Args:
        model_paths: list of paths to model .zip files.
        env_factory: callable that returns a fresh Gymnasium environment.
        n_eval_episodes: number of episodes per model.

    Returns:
        list of dicts with model_path, algo, mean_reward, std_reward.
    """
    results = []
    for path in model_paths:
        model, algo = load_model(path)
        if model is None:
            results.append({"model_path": path, "algo": None, "error": "Could not load"})
            continue

        env = env_factory()
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
        env.close()

        results.append({
            "model_path": os.path.basename(path),
            "algo": algo,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        })
        print(f"{os.path.basename(path)} ({algo}): {mean_reward:.2f} +/- {std_reward:.2f}")

    return results
