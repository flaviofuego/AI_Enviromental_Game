"""Callback for analyzing agent behavior during training."""
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class BehaviorAnalysisCallback(BaseCallback):
    """Tracks and analyzes agent movement patterns during training."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.action_history = deque(maxlen=5000)
        self.episode_rewards = deque(maxlen=100)

    def _on_step(self):
        if hasattr(self.locals, 'actions') and self.locals.get('actions') is not None:
            action = self.locals['actions'][0]
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.ndim == 0 else int(action[0])
            else:
                action = int(action)
            self.action_history.append(action)

        if self.n_calls % 25000 == 0 and len(self.action_history) > 1000:
            self._analyze_behavior()
        return True

    def _analyze_behavior(self):
        actions = list(self.action_history)
        total = len(actions)
        counts = {i: sum(1 for a in actions if a == i) for i in range(9)}

        vertical_pct = ((counts[0] + counts[1]) / total) * 100
        horizontal_pct = ((counts[2] + counts[3]) / total) * 100
        diagonal_pct = ((counts[5] + counts[6] + counts[7] + counts[8]) / total) * 100
        stay_pct = (counts[4] / total) * 100

        if self.verbose > 0:
            labels = ["Up", "Down", "Left", "Right", "Stay",
                      "UpLeft", "UpRight", "DownLeft", "DownRight"]
            print(f"\n{'=' * 50}")
            print(f"BEHAVIOR ANALYSIS - Step {self.n_calls}")
            print(f"{'=' * 50}")
            for i, label in enumerate(labels):
                print(f"{label:>10}: {counts[i]:>4} ({(counts[i] / total) * 100:>5.1f}%)")
            print(f"Vertical: {vertical_pct:.1f}% | Horizontal: {horizontal_pct:.1f}% | Diagonal: {diagonal_pct:.1f}% | Stay: {stay_pct:.1f}%")
            if vertical_pct + diagonal_pct < 10:
                print("CRITICAL: Very low vertical/diagonal movement!")
            elif vertical_pct + diagonal_pct > 15:
                print("OK: Good vertical movement balance")
