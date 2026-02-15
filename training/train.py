"""
Unified training entry point.

Usage:
    python -m training.train --preset v2_standard --env base
    python -m training.train --preset v2_quick --env powerups --n-envs 4
    python -m training.train --preset v2_quick --timesteps 1000000 --epochs 15
    python -m training.train --preset v2_quick --lr 0.0003 --gamma 0.999
"""
import os
import sys
import json
import time
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.configs.ppo_configs import PRESETS, PPOConfig
from training.callbacks import DifficultyProgressionCallback, BehaviorAnalysisCallback, MovementBalanceCallback


def create_env(env_type: str = "base"):
    """Create a training environment."""
    if env_type == "base":
        from training.envs.base_env import AirHockeyEnv
        return AirHockeyEnv()
    elif env_type == "powerups":
        from training.envs.powerups_env import AirHockeyWithPowerUpsEnv
        return AirHockeyWithPowerUpsEnv()
    else:
        raise ValueError(f"Unknown env type: {env_type}. Available: base, powerups")


def _make_env_fn(env_type: str):
    """Return a callable that creates a new env (for make_vec_env)."""
    def _init():
        return create_env(env_type)
    return _init


def _apply_cli_overrides(config: PPOConfig, args) -> PPOConfig:
    """Apply any CLI overrides on top of the preset config."""
    if args.timesteps is not None:
        config.total_timesteps = args.timesteps
    if args.epochs is not None:
        config.n_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.gamma is not None:
        config.gamma = args.gamma
    if args.ent_coef is not None:
        config.ent_coef = args.ent_coef
    if args.clip_range is not None:
        config.clip_range = args.clip_range
    if args.n_steps is not None:
        config.n_steps = args.n_steps
    if args.checkpoint_freq is not None:
        config.checkpoint_freq = args.checkpoint_freq
    if args.eval_freq is not None:
        config.eval_freq = args.eval_freq
    return config


def _save_metadata(models_dir: str, config: PPOConfig, env_type: str,
                   n_envs: int, best_mean_reward: float = None,
                   training_time: float = None):
    """Save training metadata alongside the model for smart loading later."""
    meta = {
        "preset": config.name,
        "env_type": env_type,
        "n_envs": n_envs,
        "total_timesteps": config.total_timesteps,
        "n_epochs": config.n_epochs,
        "learning_rate": str(config.learning_rate),
        "batch_size": config.batch_size,
        "gamma": config.gamma,
        "ent_coef": config.ent_coef,
        "clip_range": config.clip_range,
        "n_steps": config.n_steps,
        "net_arch_pi": list(config.net_arch_pi),
        "net_arch_vf": list(config.net_arch_vf),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if best_mean_reward is not None:
        meta["best_mean_reward"] = best_mean_reward
    if training_time is not None:
        meta["training_time_seconds"] = round(training_time, 1)
    path = os.path.join(models_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {path}")


def train(preset_name: str = "standard", env_type: str = "base",
          model_name: str = None, resume_path: str = None,
          n_envs: int = 1, args=None):
    """Run training with given preset and environment."""
    # Setup
    torch.set_num_threads(6)
    config = PRESETS.get(preset_name)
    if config is None:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

    # Apply CLI overrides
    if args is not None:
        config = _apply_cli_overrides(config, args)

    if model_name is None:
        model_name = f"air_hockey_{preset_name}_{env_type}"

    models_dir = os.path.join(project_root, "models", model_name)
    logs_dir = os.path.join(project_root, "logs", model_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"Training Configuration: {preset_name}")
    print(f"Environment: {env_type}")
    print(f"Parallel envs: {n_envs}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Epochs per update: {config.n_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gamma: {config.gamma}")
    print(f"Entropy coef: {config.ent_coef}")
    print(f"Network: pi={list(config.net_arch_pi)} vf={list(config.net_arch_vf)}")
    print(f"Model output: {models_dir}")
    print(f"{'=' * 60}")

    # Create environments
    if n_envs > 1:
        env = make_vec_env(_make_env_fn(env_type), n_envs=n_envs)
    else:
        env = Monitor(create_env(env_type))
    eval_env = create_env(env_type)

    # Create or load model
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        model = PPO.load(resume_path, env=env)
    else:
        sb3_kwargs = config.to_sb3_kwargs()
        model = PPO("MlpPolicy", env, tensorboard_log=logs_dir, **sb3_kwargs)

    # Setup callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=models_dir,
        name_prefix=model_name,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best_model"),
        log_path=logs_dir,
        eval_freq=config.eval_freq,
        deterministic=True,
        render=False,
    )
    difficulty_cb = DifficultyProgressionCallback(eval_env, eval_freq=config.eval_freq)
    behavior_cb = BehaviorAnalysisCallback()
    balance_cb = MovementBalanceCallback()

    callbacks = CallbackList([checkpoint_cb, eval_cb, difficulty_cb, behavior_cb, balance_cb])

    # Train
    t0 = time.time()
    print("Starting training...")
    model.learn(total_timesteps=config.total_timesteps, callback=callbacks, progress_bar=True)
    training_time = time.time() - t0

    # Save final model
    final_path = os.path.join(models_dir, f"{model_name}_final")
    model.save(final_path)
    print(f"Final model saved to: {final_path}.zip")

    # Read best mean reward from eval callback
    best_reward = eval_cb.best_mean_reward if hasattr(eval_cb, "best_mean_reward") else None

    # Save metadata
    _save_metadata(models_dir, config, env_type, n_envs,
                   best_mean_reward=best_reward,
                   training_time=training_time)

    env.close()
    eval_env.close()
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Train Air Hockey RL Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m training.train --preset v2_quick --env base
  python -m training.train --preset v2_standard --env powerups --n-envs 4
  python -m training.train --preset v2_quick --timesteps 2000000 --epochs 20
  python -m training.train --preset v2_quick --lr 0.0001 --gamma 0.999
        """,
    )
    # Core
    parser.add_argument("--preset", type=str, default="v2_standard",
                        choices=list(PRESETS.keys()),
                        help="Training preset (default: v2_standard)")
    parser.add_argument("--env", type=str, default="base",
                        choices=["base", "powerups"],
                        help="Environment type (default: base)")
    parser.add_argument("--name", type=str, default=None,
                        help="Custom model name")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model to resume training from")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments")

    # Hyperparameter overrides
    hp = parser.add_argument_group("Hyperparameter overrides (override preset values)")
    hp.add_argument("--timesteps", type=int, default=None,
                    help="Total training timesteps")
    hp.add_argument("--epochs", type=int, default=None,
                    help="PPO epochs per update")
    hp.add_argument("--lr", type=float, default=None,
                    help="Learning rate (constant, overrides schedule)")
    hp.add_argument("--batch-size", type=int, default=None,
                    help="Mini-batch size")
    hp.add_argument("--gamma", type=float, default=None,
                    help="Discount factor")
    hp.add_argument("--ent-coef", type=float, default=None,
                    help="Entropy coefficient")
    hp.add_argument("--clip-range", type=float, default=None,
                    help="PPO clipping range")
    hp.add_argument("--n-steps", type=int, default=None,
                    help="Steps per rollout collection")
    hp.add_argument("--checkpoint-freq", type=int, default=None,
                    help="Checkpoint save frequency (steps)")
    hp.add_argument("--eval-freq", type=int, default=None,
                    help="Evaluation frequency (steps)")

    args = parser.parse_args()
    train(args.preset, args.env, args.name, args.resume, args.n_envs, args=args)


if __name__ == "__main__":
    main()
