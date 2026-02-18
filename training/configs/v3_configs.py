"""
v3 Training presets — optimized architecture configurations.

Improvements over v2:
  - Warmup + cosine LR schedule for smoother convergence.
  - Configurable env wrappers (VecNormalize, VecFrameStack).
  - Extended observation space (opponent velocity + goal distances).
  - Separate network widths for policy vs value.
  - DQN presets for comparison.
  - GAE lambda tuning for better advantage estimation.
"""
from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from training.configs.schedules import (
    cosine_warmup_schedule,
    cosine_schedule,
    linear_schedule,
)
from training.configs.ppo_configs import PPOConfig, PRESETS


# ─────────────────────────────────────────────────────────────────────
# Extended config with v3 features
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig(PPOConfig):
    """Extended training configuration with env-wrapper and algorithm options.

    Inherits all PPOConfig fields and adds:
      - Algorithm selection (PPO / DQN).
      - Env-wrapper pipeline configuration.
      - Extended observation flags.
      - Orthogonal initialization flag.
    """

    # Algorithm
    algorithm: str = "PPO"  # "PPO" | "DQN"

    # Env wrappers (applied in this order)
    use_vec_normalize: bool = False
    normalize_observations: bool = True
    normalize_rewards: bool = True
    vec_normalize_kwargs: Dict[str, Any] = field(default_factory=dict)

    use_frame_stack: bool = False
    n_frame_stack: int = 4

    # Extended observation (17D vs 13D)
    use_extended_obs: bool = False

    # Powerup system configuration
    powerup_phases: list[int] = field(default_factory=lambda: [1, 2, 3])
    """Powerup phases to activate when env_type='powerups'. Default: phases 1-3."""

    # Policy kwargs extras
    ortho_init: bool = True
    normalize_advantage: bool = True

    # DQN-specific (unused if algorithm == "PPO")
    dqn_buffer_size: int = 100_000
    dqn_learning_starts: int = 10_000
    dqn_tau: float = 0.005
    dqn_train_freq: int = 4
    dqn_gradient_steps: int = 1
    dqn_double_q: bool = True

    def to_sb3_kwargs(self) -> Dict[str, Any]:
        """Convert to SB3 constructor kwargs for the selected algorithm."""
        if self.algorithm == "DQN":
            return self._to_dqn_kwargs()
        return self._to_ppo_kwargs()

    def _to_ppo_kwargs(self) -> Dict[str, Any]:
        """PPO-specific kwargs with v3 extensions."""
        kwargs = super().to_sb3_kwargs()

        # Inject ortho_init into policy_kwargs
        kwargs["policy_kwargs"]["ortho_init"] = self.ortho_init

        # normalize_advantage is a PPO constructor kwarg
        kwargs["normalize_advantage"] = self.normalize_advantage

        return kwargs

    def _to_dqn_kwargs(self) -> Dict[str, Any]:
        """DQN-specific kwargs."""
        return {
            "learning_rate": self.learning_rate,
            "buffer_size": self.dqn_buffer_size,
            "learning_starts": self.dqn_learning_starts,
            "batch_size": self.batch_size,
            "tau": self.dqn_tau,
            "gamma": self.gamma,
            "train_freq": self.dqn_train_freq,
            "gradient_steps": self.dqn_gradient_steps,
            "policy_kwargs": {
                "activation_fn": self.activation_fn,
                "net_arch": list(self.net_arch_pi),  # DQN uses single net
            },
            "verbose": 1,
        }


# ─────────────────────────────────────────────────────────────────────
# v3 Presets
# ─────────────────────────────────────────────────────────────────────

V3_QUICK = TrainingConfig(
    name="v3_quick",
    algorithm="PPO",
    total_timesteps=800_000,
    learning_rate=cosine_warmup_schedule(peak=3e-4, warmup_frac=0.08, final=5e-6),
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.015,
    target_kl=0.025,
    net_arch_pi=[256, 128],
    net_arch_vf=[256, 128],
    ortho_init=True,
    normalize_advantage=True,
    checkpoint_freq=100_000,
    eval_freq=25_000,
    # Env wrappers — quick doesn't need heavy normalization
    use_vec_normalize=False,
    use_frame_stack=False,
    use_extended_obs=False,
)

V3_OPTIMIZED = TrainingConfig(
    name="v3_optimized",
    algorithm="PPO",
    total_timesteps=2_000_000,
    learning_rate=cosine_warmup_schedule(peak=3e-4, warmup_frac=0.1, final=1e-6),
    n_steps=4096,
    batch_size=256,
    n_epochs=10,
    gamma=0.997,
    gae_lambda=0.98,
    clip_range=0.2,
    ent_coef=0.02,  # Higher exploration initially
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.025,
    # Wider policy net, narrower value net
    net_arch_pi=[256, 256],
    net_arch_vf=[128, 128],
    activation_fn=torch.nn.ReLU,
    ortho_init=True,
    normalize_advantage=True,
    # Env wrappers
    use_vec_normalize=True,
    normalize_observations=True,
    normalize_rewards=True,
    use_frame_stack=False,
    use_extended_obs=True,
    checkpoint_freq=200_000,
    eval_freq=50_000,
)

V3_DEEP = TrainingConfig(
    name="v3_deep",
    algorithm="PPO",
    total_timesteps=5_000_000,
    learning_rate=cosine_warmup_schedule(peak=2e-4, warmup_frac=0.05, final=5e-7),
    n_steps=4096,
    batch_size=256,
    n_epochs=15,
    gamma=0.997,
    gae_lambda=0.98,
    clip_range=0.15,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.02,
    net_arch_pi=[512, 256, 128],
    net_arch_vf=[256, 128],
    activation_fn=torch.nn.ReLU,
    ortho_init=True,
    normalize_advantage=True,
    use_vec_normalize=True,
    normalize_observations=True,
    normalize_rewards=True,
    use_frame_stack=False,
    use_extended_obs=True,
    checkpoint_freq=250_000,
    eval_freq=75_000,
)

V3_FRAMESTACK = TrainingConfig(
    name="v3_framestack",
    algorithm="PPO",
    total_timesteps=2_000_000,
    learning_rate=cosine_warmup_schedule(peak=3e-4, warmup_frac=0.1, final=1e-6),
    n_steps=4096,
    batch_size=256,
    n_epochs=10,
    gamma=0.997,
    gae_lambda=0.98,
    clip_range=0.2,
    ent_coef=0.02,
    target_kl=0.025,
    # Wider net to handle temporal input (obs_dim * n_stack)
    net_arch_pi=[256, 256],
    net_arch_vf=[128, 128],
    ortho_init=True,
    normalize_advantage=True,
    use_vec_normalize=True,
    use_frame_stack=True,
    n_frame_stack=4,
    use_extended_obs=True,
    checkpoint_freq=200_000,
    eval_freq=50_000,
)

V3_DQN = TrainingConfig(
    name="v3_dqn",
    algorithm="DQN",
    total_timesteps=2_000_000,
    learning_rate=cosine_schedule(1e-4, 1e-6),
    batch_size=64,
    gamma=0.997,
    net_arch_pi=[256, 256],
    net_arch_vf=[256, 256],  # unused by DQN
    activation_fn=torch.nn.ReLU,
    # DQN-specific
    dqn_buffer_size=200_000,
    dqn_learning_starts=20_000,
    dqn_tau=0.005,
    dqn_train_freq=4,
    dqn_gradient_steps=1,
    dqn_double_q=True,
    # Env
    use_vec_normalize=True,
    use_extended_obs=True,
    checkpoint_freq=200_000,
    eval_freq=50_000,
)

V3_EXPLORATION = TrainingConfig(
    name="v3_exploration",
    algorithm="PPO",
    total_timesteps=1_500_000,
    learning_rate=cosine_warmup_schedule(peak=3e-4, warmup_frac=0.15, final=5e-6),
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.25,
    ent_coef=0.03,  # High entropy for maximum exploration
    target_kl=0.04,
    net_arch_pi=[256, 256, 128],
    net_arch_vf=[256, 128],
    ortho_init=True,
    normalize_advantage=True,
    use_vec_normalize=False,
    use_frame_stack=False,
    use_extended_obs=False,
    checkpoint_freq=150_000,
    eval_freq=30_000,
)


# Register v3 presets
V3_PRESETS: Dict[str, TrainingConfig] = {
    "v3_quick": V3_QUICK,
    "v3_optimized": V3_OPTIMIZED,
    "v3_deep": V3_DEEP,
    "v3_framestack": V3_FRAMESTACK,
    "v3_dqn": V3_DQN,
    "v3_exploration": V3_EXPLORATION,
}

# Add to global presets for CLI compatibility
PRESETS.update(V3_PRESETS)
