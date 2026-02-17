"""
Composable environment wrapping pipeline for training.

Applies wrappers (Monitor, VecNormalize, VecFrameStack) based on
``TrainingConfig`` flags. Keeps ``train.py`` clean (SRP).

Usage:
    from training.env_pipeline import build_training_envs

    env, eval_env = build_training_envs(config, env_type="base", n_envs=4)
"""
from __future__ import annotations

import os
from typing import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
)

from training.envs.observation_builder import ObservationBuilder


def _create_base_env(env_type: str, obs_builder: ObservationBuilder | None = None):
    """Create a single training environment instance."""
    if env_type == "base":
        from training.envs.base_env import AirHockeyEnv
        return AirHockeyEnv(obs_builder=obs_builder)
    elif env_type == "powerups":
        from training.envs.powerups_env import AirHockeyWithPowerUpsEnv
        return AirHockeyWithPowerUpsEnv()
    else:
        raise ValueError(f"Unknown env type: {env_type}. Available: base, powerups")


def _make_env_fn(env_type: str, obs_builder: ObservationBuilder | None = None) -> Callable:
    """Return a closure that creates a new env (for ``make_vec_env``)."""
    def _init():
        return _create_base_env(env_type, obs_builder)
    return _init


def _resolve_obs_builder(config) -> ObservationBuilder | None:
    """Determine the ObservationBuilder from a config, if applicable."""
    if not hasattr(config, "use_extended_obs"):
        return None
    if config.use_extended_obs:
        return ObservationBuilder.extended()
    return ObservationBuilder.standard()


def build_training_envs(
    config,
    env_type: str = "base",
    n_envs: int = 1,
    log_dir: str | None = None,
) -> tuple[VecEnv | gym.Env, gym.Env]:
    """Build training and eval environments with the appropriate wrappers.

    Returns:
        (train_env, eval_env) — train_env is vectorized if ``n_envs > 1``
        or if wrappers require it.
    """
    obs_builder = _resolve_obs_builder(config)

    # Determine if we need a VecEnv (wrappers like VecNormalize require it)
    needs_vec = (
        n_envs > 1
        or getattr(config, "use_vec_normalize", False)
        or getattr(config, "use_frame_stack", False)
    )

    # ── Training env ─────────────────────────────────────────────
    if needs_vec:
        train_env = make_vec_env(
            _make_env_fn(env_type, obs_builder),
            n_envs=max(n_envs, 1),
        )
    else:
        train_env = DummyVecEnv([_make_env_fn(env_type, obs_builder)])

    # ── Eval env (always single, unwrapped for score access) ─────
    eval_env_raw = _create_base_env(env_type, obs_builder)

    # ── VecNormalize ─────────────────────────────────────────────
    if getattr(config, "use_vec_normalize", False):
        norm_kwargs = getattr(config, "vec_normalize_kwargs", {})
        train_env = VecNormalize(
            train_env,
            norm_obs=getattr(config, "normalize_observations", True),
            norm_reward=getattr(config, "normalize_rewards", True),
            clip_obs=10.0,
            clip_reward=10.0,
            **norm_kwargs,
        )
        # Eval env uses same normalization stats but doesn't update them
        eval_env = DummyVecEnv([lambda: eval_env_raw])
        eval_env = VecNormalize(
            eval_env,
            norm_obs=getattr(config, "normalize_observations", True),
            norm_reward=False,  # Don't normalize eval rewards
            clip_obs=10.0,
            training=False,
        )
    else:
        eval_env = eval_env_raw

    # ── VecFrameStack ────────────────────────────────────────────
    if getattr(config, "use_frame_stack", False):
        n_stack = getattr(config, "n_frame_stack", 4)
        train_env = VecFrameStack(train_env, n_stack=n_stack)
        if isinstance(eval_env, VecEnv):
            eval_env = VecFrameStack(eval_env, n_stack=n_stack)
        else:
            # Wrap raw eval env in VecEnv first, then stack
            eval_env = DummyVecEnv([lambda: eval_env])
            eval_env = VecFrameStack(eval_env, n_stack=n_stack)

    return train_env, eval_env


def save_vec_normalize(env: VecEnv, path: str) -> bool:
    """Save VecNormalize stats if the env uses it.

    Returns True if stats were saved.
    """
    # Walk the wrapper chain to find VecNormalize
    current = env
    while current is not None:
        if isinstance(current, VecNormalize):
            current.save(path)
            return True
        current = getattr(current, "venv", None)
    return False


def sync_vec_normalize(train_env: VecEnv, eval_env: VecEnv) -> None:
    """Copy normalization stats from train env to eval env.

    Call this before evaluation to ensure consistent normalization.
    """
    train_norm = _find_vec_normalize(train_env)
    eval_norm = _find_vec_normalize(eval_env)
    if train_norm is not None and eval_norm is not None:
        eval_norm.obs_rms = train_norm.obs_rms
        eval_norm.ret_rms = train_norm.ret_rms


def _find_vec_normalize(env: VecEnv) -> VecNormalize | None:
    """Walk wrapper chain to find VecNormalize layer."""
    current = env
    while current is not None:
        if isinstance(current, VecNormalize):
            return current
        current = getattr(current, "venv", None)
    return None
