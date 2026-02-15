"""Callback for adaptive difficulty progression during training."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DifficultyProgressionCallback(BaseCallback):
    """Adjusts opponent difficulty based on agent performance."""

    def __init__(self, eval_env, eval_freq=50000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -float('inf')

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            obs, _ = self.eval_env.reset()
            for _ in range(5):
                done = False
                ep_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = self.eval_env.step(action)
                    done = done or truncated
                    ep_reward += reward
                rewards.append(ep_reward)
                obs, _ = self.eval_env.reset()

            mean_reward = np.mean(rewards)
            if mean_reward > self.best_mean_reward + 0.5:
                self.best_mean_reward = mean_reward
                unwrapped = self.eval_env.unwrapped
                if hasattr(unwrapped, 'increase_opponent_difficulty'):
                    unwrapped.increase_opponent_difficulty(mean_reward)
                if self.verbose > 0:
                    print(f"Step {self.n_calls}: Mean reward: {mean_reward:.2f}, increasing difficulty")
            elif self.verbose > 0:
                print(f"Step {self.n_calls}: Mean reward: {mean_reward:.2f}, maintaining difficulty")
        return True
