"""
Learning rate and hyperparameter schedules for RL training.

Reusable schedule functions compatible with Stable Baselines3's
``progress_remaining`` convention (1.0 → 0.0 as training progresses).

Usage:
    from training.configs.schedules import cosine_warmup_schedule, linear_schedule

    config = PPOConfig(
        learning_rate=cosine_warmup_schedule(peak=3e-4, warmup_frac=0.1, final=1e-6),
    )
"""
from __future__ import annotations

import math
from typing import Callable

# Type alias for SB3-compatible schedule functions.
# SB3 passes ``progress_remaining`` ∈ [1.0, 0.0] (decreasing over training).
Schedule = Callable[[float], float]


def linear_schedule(initial: float, final: float = 1e-5) -> Schedule:
    """Linearly interpolate from *initial* → *final* over training.

    Args:
        initial: Starting value (at progress_remaining=1.0).
        final: Ending value (at progress_remaining=0.0).
    """
    def _schedule(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining
    _schedule.__name__ = f"linear({initial:.1e}→{final:.1e})"
    _schedule.__qualname__ = _schedule.__name__
    return _schedule


def cosine_schedule(initial: float, final: float = 1e-6) -> Schedule:
    """Cosine annealing from *initial* → *final*.

    Smoother than linear — decays slowly at start, then faster in the middle,
    and gently approaches *final*.

    Args:
        initial: Starting value.
        final: Minimum value at end of training.
    """
    def _schedule(progress_remaining: float) -> float:
        # progress_remaining: 1→0, so progress_done: 0→1
        progress_done = 1.0 - progress_remaining
        return final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * progress_done))
    _schedule.__name__ = f"cosine({initial:.1e}→{final:.1e})"
    _schedule.__qualname__ = _schedule.__name__
    return _schedule


def cosine_warmup_schedule(
    peak: float,
    final: float = 1e-6,
    warmup_frac: float = 0.1,
) -> Schedule:
    """Warmup + cosine annealing schedule.

    1. **Warmup phase** (first ``warmup_frac`` of training):
       Linearly ramp from ``final`` → ``peak``.
    2. **Cosine decay phase** (remaining training):
       Cosine anneal from ``peak`` → ``final``.

    Args:
        peak: Maximum LR reached at end of warmup.
        final: Minimum LR at end of training (and start of warmup).
        warmup_frac: Fraction of training used for warmup (0.0–1.0).
    """
    def _schedule(progress_remaining: float) -> float:
        progress_done = 1.0 - progress_remaining  # 0 → 1

        if progress_done < warmup_frac:
            # Linear warmup
            warmup_progress = progress_done / warmup_frac
            return final + (peak - final) * warmup_progress
        else:
            # Cosine decay from peak to final
            decay_progress = (progress_done - warmup_frac) / (1.0 - warmup_frac)
            return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * decay_progress))

    _schedule.__name__ = f"warmup_cosine(peak={peak:.1e}, warmup={warmup_frac})"
    _schedule.__qualname__ = _schedule.__name__
    return _schedule


def constant_schedule(value: float) -> Schedule:
    """Constant value throughout training."""
    def _schedule(progress_remaining: float) -> float:
        return value
    _schedule.__name__ = f"constant({value:.1e})"
    _schedule.__qualname__ = _schedule.__name__
    return _schedule


def stepped_schedule(
    steps: list[tuple[float, float]],
    initial: float | None = None,
) -> Schedule:
    """Step-wise schedule with discrete drops at specified progress thresholds.

    Args:
        steps: List of ``(progress_done_threshold, value)`` tuples,
               sorted ascending by threshold. When ``progress_done`` passes
               a threshold, value switches to the corresponding entry.
        initial: Value before the first threshold. Defaults to ``steps[0][1]``.

    Example:
        >>> sched = stepped_schedule([(0.0, 3e-4), (0.5, 1e-4), (0.8, 3e-5)])
    """
    if initial is None:
        initial = steps[0][1] if steps else 1e-4

    # Sort ascending by threshold
    sorted_steps = sorted(steps, key=lambda s: s[0])

    def _schedule(progress_remaining: float) -> float:
        progress_done = 1.0 - progress_remaining
        value = initial
        for threshold, step_value in sorted_steps:
            if progress_done >= threshold:
                value = step_value
            else:
                break
        return value

    _schedule.__name__ = f"stepped({len(sorted_steps)} steps)"
    _schedule.__qualname__ = _schedule.__name__
    return _schedule


def exponential_schedule(initial: float, final: float = 1e-6) -> Schedule:
    """Exponential decay from *initial* → *final*.

    Decays quickly at first, then slows down — useful when you want
    aggressive early exploration followed by fine-tuning.
    """
    ratio = final / max(initial, 1e-12)

    def _schedule(progress_remaining: float) -> float:
        progress_done = 1.0 - progress_remaining
        return initial * (ratio ** progress_done)

    _schedule.__name__ = f"exponential({initial:.1e}→{final:.1e})"
    _schedule.__qualname__ = _schedule.__name__
    return _schedule
