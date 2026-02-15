"""
Training analysis utilities — plot reward curves, action distributions, etc.
"""
import os
import numpy as np


def load_monitor_data(log_dir: str):
    """
    Load training data from Monitor CSV files.
    Returns a dict with reward and episode length lists.
    """
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    try:
        results = load_results(log_dir)
        x, y = ts2xy(results, "timesteps")
        return {"timesteps": x.tolist(), "rewards": y.tolist()}
    except Exception as e:
        print(f"Error loading monitor data: {e}")
        return {"timesteps": [], "rewards": []}


def print_training_summary(log_dir: str):
    """Print a summary of the training run."""
    data = load_monitor_data(log_dir)
    if not data["rewards"]:
        print("No training data found.")
        return

    rewards = np.array(data["rewards"])
    print(f"Total episodes: {len(rewards)}")
    print(f"Mean reward: {rewards.mean():.2f}")
    print(f"Max reward: {rewards.max():.2f}")
    print(f"Min reward: {rewards.min():.2f}")
    print(f"Last 100 episodes mean: {rewards[-100:].mean():.2f}")
