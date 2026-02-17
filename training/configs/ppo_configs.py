"""
Centralized PPO configuration presets for training.
Consolidates the 4 different configs from the training scripts.
"""
import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable


def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    """Linear learning rate schedule from initial_value to final_value."""
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    func.__name__ = f"linear({initial_value:.1e}→{final_value:.1e})"
    func.__qualname__ = func.__name__
    return func


@dataclass
class PPOConfig:
    """PPO hyperparameter configuration."""
    name: str = "standard"
    total_timesteps: int = 1_500_000
    learning_rate: Any = None  # float or schedule
    n_steps: int = 4096
    batch_size: int = 128
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.98
    clip_range: float = 0.15
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Network architecture
    net_arch_pi: list = field(default_factory=lambda: [256, 256, 128])
    net_arch_vf: list = field(default_factory=lambda: [256, 256, 128])
    activation_fn: Any = field(default_factory=lambda: torch.nn.ReLU)
    # Training stability
    target_kl: Optional[float] = None
    # Checkpointing
    checkpoint_freq: int = 200_000
    eval_freq: int = 50_000

    def __post_init__(self):
        if self.learning_rate is None:
            self.learning_rate = linear_schedule(3e-4, 1e-5)

    def to_sb3_kwargs(self) -> Dict[str, Any]:
        """Convert to stable-baselines3 PPO constructor kwargs."""
        kwargs = {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "policy_kwargs": {
                "activation_fn": self.activation_fn,
                "net_arch": {
                    "pi": list(self.net_arch_pi),
                    "vf": list(self.net_arch_vf),
                },
            },
            "verbose": 1,
        }
        if self.target_kl is not None:
            kwargs["target_kl"] = self.target_kl
        return kwargs


# --- Presets ---

QUICK_TRAIN = PPOConfig(
    name="quick",
    total_timesteps=500_000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    net_arch_pi=[128, 128],
    net_arch_vf=[128, 128],
    checkpoint_freq=100_000,
    eval_freq=25_000,
)

STANDARD_TRAIN = PPOConfig(
    name="standard",
    total_timesteps=1_500_000,
    # learning_rate uses default linear schedule
    n_steps=4096,
    batch_size=128,
    gamma=0.995,
    gae_lambda=0.98,
    clip_range=0.15,
    ent_coef=0.01,
    target_kl=0.03,
    net_arch_pi=[256, 256, 128],
    net_arch_vf=[256, 256, 128],
    checkpoint_freq=200_000,
    eval_freq=50_000,
)

DEEP_TRAIN = PPOConfig(
    name="deep",
    total_timesteps=3_000_000,
    # learning_rate uses default linear schedule
    n_steps=4096,
    batch_size=128,
    gamma=0.995,
    gae_lambda=0.98,
    clip_range=0.15,
    ent_coef=0.005,
    target_kl=0.02,
    net_arch_pi=[256, 256, 128],
    net_arch_vf=[256, 256, 128],
    checkpoint_freq=200_000,
    eval_freq=75_000,
)

EXPLORATION_TRAIN = PPOConfig(
    name="exploration",
    total_timesteps=1_500_000,
    n_steps=2048,
    batch_size=128,
    gamma=0.99,
    ent_coef=0.02,  # Higher entropy for more exploration
    clip_range=0.2,
    net_arch_pi=[256, 256, 128],
    net_arch_vf=[256, 256, 128],
    checkpoint_freq=200_000,
    eval_freq=50_000,
)

PRESETS = {
    "quick": QUICK_TRAIN,
    "standard": STANDARD_TRAIN,
    "deep": DEEP_TRAIN,
    "exploration": EXPLORATION_TRAIN,
}

# ─── v2 Presets (aligned with new 9-action env + hockey reward) ───

V2_QUICK = PPOConfig(
    name="v2_quick",
    total_timesteps=800_000,
    learning_rate=linear_schedule(3e-4, 5e-6),
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.015,  # Slightly higher to explore 9 actions
    net_arch_pi=[256, 128],
    net_arch_vf=[256, 128],
    checkpoint_freq=100_000,
    eval_freq=25_000,
)

V2_STANDARD = PPOConfig(
    name="v2_standard",
    total_timesteps=3_000_000,
    learning_rate=linear_schedule(3e-4, 1e-5),
    n_steps=4096,
    batch_size=256,
    n_epochs=10,
    gamma=0.997,
    gae_lambda=0.98,
    clip_range=0.15,
    ent_coef=0.015,
    target_kl=0.03,
    net_arch_pi=[512, 256, 128],
    net_arch_vf=[512, 256, 128],
    checkpoint_freq=200_000,
    eval_freq=50_000,
)

V2_DEEP = PPOConfig(
    name="v2_deep",
    total_timesteps=5_000_000,
    learning_rate=linear_schedule(2e-4, 5e-6),
    n_steps=4096,
    batch_size=256,
    n_epochs=15,
    gamma=0.997,
    gae_lambda=0.98,
    clip_range=0.12,
    ent_coef=0.01,
    target_kl=0.02,
    net_arch_pi=[512, 256, 128],
    net_arch_vf=[512, 256, 128],
    checkpoint_freq=250_000,
    eval_freq=100_000,
)

PRESETS.update({
    "v2_quick": V2_QUICK,
    "v2_standard": V2_STANDARD,
    "v2_deep": V2_DEEP,
})
