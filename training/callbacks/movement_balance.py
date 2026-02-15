"""Callback for monitoring movement balance during training."""
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class MovementBalanceCallback(BaseCallback):
    """Monitors action distribution and warns about movement imbalance."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.action_history = deque(maxlen=5000)

    def _on_step(self):
        if self.locals.get('actions') is not None:
            action = self.locals['actions'][0]
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.ndim == 0 else int(action[0])
            else:
                action = int(action)
            self.action_history.append(action)

        if self.n_calls % 25000 == 0 and len(self.action_history) > 1000:
            self._check_balance()
        return True

    def _check_balance(self):
        actions = list(self.action_history)
        total = len(actions)
        # Pure vertical + diagonal (which includes vertical component)
        vertical = sum(1 for a in actions if a in (0, 1, 5, 6, 7, 8))
        vertical_pct = (vertical / total) * 100

        if self.verbose > 0 and vertical_pct < 15:
            print(f"WARNING: Low vertical movement ({vertical_pct:.1f}%) at step {self.n_calls}")
