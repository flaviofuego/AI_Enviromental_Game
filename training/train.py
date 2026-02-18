"""
Hockey RL Training CLI — Powered by Typer + Rich.

Commands:
    start    Entrenar con preset y overrides opcionales
    wizard   Asistente interactivo de configuración visual
    presets  Listar presets disponibles con sus configuraciones

Usage:
    uv run python -m training.train start --preset v3_optimized --n-envs 4
    uv run python -m training.train wizard
    uv run python -m training.train presets
"""
from __future__ import annotations

import copy
import os
import sys
import json
import time
from typing import Annotated, Optional

import typer
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.rule import Rule
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich import box

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.configs.ppo_configs import PRESETS, PPOConfig
from training.configs.v3_configs import TrainingConfig, V3_PRESETS  # registers v3 in PRESETS
from training.callbacks import (
    DifficultyProgressionCallback,
    BehaviorAnalysisCallback,
    MovementBalanceCallback,
)
from training.env_pipeline import build_training_envs, save_vec_normalize

# ─── Globals ────────────────────────────────────────────────────────
console = Console()

app = typer.Typer(
    name="hockey-train",
    help="🏒 Hockey RL Training CLI — Entrenamiento de agentes de IA",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

_ALGORITHMS = {"PPO": PPO, "DQN": DQN}


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def create_env(env_type: str = "base", powerup_phases: list[int] | None = None):
    """Create a single training environment.

    Parameters
    ----------
    env_type       : "base" | "powerups"
    powerup_phases : list of phase ints (1-8) for the powerups env.
                     Defaults to [1, 2, 3] when not specified.
    """
    if env_type == "base":
        from training.envs.base_env import AirHockeyEnv
        return AirHockeyEnv()
    elif env_type == "powerups":
        from training.envs.powerups_env import AirHockeyWithPowerUpsEnv
        phases = powerup_phases or [1, 2, 3]
        return AirHockeyWithPowerUpsEnv(phases=phases)
    else:
        raise ValueError(f"Unknown env type: {env_type}. Available: base, powerups")


def _lr_display(lr) -> str:
    """Human-readable LR representation."""
    if isinstance(lr, (int, float)):
        return f"{lr:.1e}"
    return getattr(lr, "__name__", "schedule")


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs}s"


# ═══════════════════════════════════════════════════════════════════
# Rich display functions
# ═══════════════════════════════════════════════════════════════════

def _display_config(
    preset_name: str,
    config: PPOConfig,
    env_type: str,
    n_envs: int,
    models_dir: str,
):
    """Display a rich panel with the full training configuration."""
    is_v3 = isinstance(config, TrainingConfig)
    algorithm = getattr(config, "algorithm", "PPO")

    # ── Main info ────────────────────────────────────────────────
    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column("Key", style="dim")
    info.add_column("Value", style="bold")
    info.add_row("Preset", f"[cyan]{preset_name}[/cyan]")
    info.add_row("Algoritmo", f"[magenta]{algorithm}[/magenta]")
    info.add_row("Entorno", env_type)
    info.add_row("Envs paralelos", str(n_envs))
    info.add_row("Total timesteps", f"[green]{config.total_timesteps:,}[/green]")
    info.add_row("Salida", models_dir)

    console.print()
    console.print(
        Panel(info,
              title="🏒 [bold blue]Configuración de Entrenamiento[/bold blue]",
              border_style="blue", expand=False),
    )

    # ── Hyperparameters ──────────────────────────────────────────
    hp = Table(title="[bold]Hiperparámetros[/bold]", box=box.ROUNDED)
    hp.add_column("Param", style="cyan", min_width=16)
    hp.add_column("Valor", style="yellow", justify="right")
    hp.add_row("Learning Rate", _lr_display(config.learning_rate))
    hp.add_row("Batch Size", str(config.batch_size))
    hp.add_row("N Steps", str(config.n_steps))
    hp.add_row("Epochs", str(config.n_epochs))
    hp.add_row("Gamma (γ)", str(config.gamma))
    hp.add_row("GAE Lambda (λ)", str(config.gae_lambda))
    hp.add_row("Clip Range", str(config.clip_range))
    hp.add_row("Entropy Coef", str(config.ent_coef))
    hp.add_row("VF Coef", str(config.vf_coef))
    hp.add_row("Max Grad Norm", str(config.max_grad_norm))
    if config.target_kl:
        hp.add_row("Target KL", str(config.target_kl))

    # ── Network ──────────────────────────────────────────────────
    net = Table(title="[bold]Red Neuronal[/bold]", box=box.ROUNDED)
    net.add_column("Componente", style="cyan")
    net.add_column("Arquitectura", style="magenta")
    net.add_row("Policy (π)", str(list(config.net_arch_pi)))
    net.add_row("Value (V)", str(list(config.net_arch_vf)))
    net.add_row("Activación", config.activation_fn.__name__)

    tables: list[Table] = [hp, net]

    # ── Wrappers & Obs (v3) ──────────────────────────────────────
    if is_v3:
        _on = "[green]✓[/green]"
        _off = "[dim]✗[/dim]"
        wrap = Table(title="[bold]Wrappers & Obs[/bold]", box=box.ROUNDED)
        wrap.add_column("Feature", style="cyan")
        wrap.add_column("Estado", min_width=14)
        obs_dim = "17D" if config.use_extended_obs else "13D"
        wrap.add_row("Obs Space", f"[green]{obs_dim}[/green]")
        wrap.add_row("VecNormalize", _on if config.use_vec_normalize else _off)
        fs_txt = f"{_on} (n={config.n_frame_stack})" if config.use_frame_stack else _off
        wrap.add_row("FrameStack", fs_txt)
        wrap.add_row("Ortho Init", _on if config.ortho_init else _off)
        wrap.add_row("Norm Advantage", _on if config.normalize_advantage else _off)
        tables.append(wrap)

    # ── DQN-specific ─────────────────────────────────────────────
    if is_v3 and algorithm == "DQN":
        dqn = Table(title="[bold]DQN Config[/bold]", box=box.ROUNDED)
        dqn.add_column("Param", style="cyan")
        dqn.add_column("Valor", style="yellow", justify="right")
        dqn.add_row("Buffer Size", f"{config.dqn_buffer_size:,}")
        dqn.add_row("Learning Starts", f"{config.dqn_learning_starts:,}")
        dqn.add_row("Tau", str(config.dqn_tau))
        dqn.add_row("Train Freq", str(config.dqn_train_freq))
        dqn.add_row("Double Q", _on if config.dqn_double_q else _off)
        tables.append(dqn)

    # ── Checkpointing ────────────────────────────────────────────
    ck = Table(title="[bold]Checkpointing[/bold]", box=box.ROUNDED)
    ck.add_column("Param", style="cyan")
    ck.add_column("Valor", style="yellow", justify="right")
    ck.add_row("Checkpoint freq", f"{config.checkpoint_freq:,}")
    ck.add_row("Eval freq", f"{config.eval_freq:,}")
    tables.append(ck)

    console.print(Columns(tables, padding=2))
    console.print()


def _display_results(
    model_name: str,
    models_dir: str,
    training_time: float,
    best_reward: float | None,
):
    """Display a rich panel with training results."""
    results = Table(show_header=False, box=None, padding=(0, 2))
    results.add_column("Key", style="dim")
    results.add_column("Value", style="bold green")
    results.add_row("Tiempo total", _format_duration(training_time))
    if best_reward is not None:
        results.add_row("Mejor Reward", f"{best_reward:.1f}")
    results.add_row("Modelo final", os.path.join(models_dir, f"{model_name}_final.zip"))
    results.add_row("Best model", os.path.join(models_dir, "best_model"))
    results.add_row("Metadata", os.path.join(models_dir, "metadata.json"))

    console.print(
        Panel(results,
              title="✅ [bold green]Entrenamiento Completo[/bold green]",
              border_style="green", expand=False),
    )


# ═══════════════════════════════════════════════════════════════════
# Core training logic
# ═══════════════════════════════════════════════════════════════════

def _save_metadata(
    models_dir: str,
    config: PPOConfig,
    env_type: str,
    n_envs: int,
    best_mean_reward: float | None = None,
    training_time: float | None = None,
):
    """Save training metadata alongside the model for reproducibility."""
    algorithm = getattr(config, "algorithm", "PPO")
    meta = {
        "preset": config.name,
        "algorithm": algorithm,
        "env_type": env_type,
        "n_envs": n_envs,
        "total_timesteps": config.total_timesteps,
        "n_epochs": config.n_epochs,
        "learning_rate": _lr_display(config.learning_rate),
        "batch_size": config.batch_size,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "ent_coef": config.ent_coef,
        "clip_range": config.clip_range,
        "n_steps": config.n_steps,
        "net_arch_pi": list(config.net_arch_pi),
        "net_arch_vf": list(config.net_arch_vf),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if isinstance(config, TrainingConfig):
        meta.update({
            "use_vec_normalize": config.use_vec_normalize,
            "use_frame_stack": config.use_frame_stack,
            "n_frame_stack": config.n_frame_stack if config.use_frame_stack else None,
            "use_extended_obs": config.use_extended_obs,
            "ortho_init": config.ortho_init,
            "normalize_advantage": config.normalize_advantage,
        })
    if best_mean_reward is not None:
        meta["best_mean_reward"] = best_mean_reward
    if training_time is not None:
        meta["training_time_seconds"] = round(training_time, 1)

    path = os.path.join(models_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def _create_model(config: PPOConfig, env, logs_dir: str, resume_path: str | None = None):
    """Create or resume an RL model based on config."""
    algorithm = getattr(config, "algorithm", "PPO")
    algo_cls = _ALGORITHMS.get(algorithm)
    if algo_cls is None:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(_ALGORITHMS.keys())}")

    if resume_path and os.path.exists(resume_path):
        console.print(f"[yellow]Reanudando desde:[/yellow] {resume_path}")
        return algo_cls.load(resume_path, env=env)

    sb3_kwargs = config.to_sb3_kwargs()
    return algo_cls("MlpPolicy", env, tensorboard_log=logs_dir, **sb3_kwargs)


def _build_callbacks(config, eval_env, models_dir, logs_dir, model_name):
    """Assemble the callback list for training."""
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
    cbs = [checkpoint_cb, eval_cb]

    algorithm = getattr(config, "algorithm", "PPO")
    if algorithm == "PPO":
        cbs.append(DifficultyProgressionCallback(eval_env, eval_freq=config.eval_freq))
        cbs.append(BehaviorAnalysisCallback())
        cbs.append(MovementBalanceCallback())

    return CallbackList(cbs), eval_cb


def run_training(
    config: PPOConfig,
    env_type: str = "base",
    model_name: str | None = None,
    resume_path: str | None = None,
    n_envs: int = 1,
) -> str:
    """Execute training with given config. Returns path to final model."""
    torch.set_num_threads(6)

    preset_name = config.name
    if model_name is None:
        model_name = f"air_hockey_{preset_name}_{env_type}"

    models_dir = os.path.join(project_root, "models", model_name)
    logs_dir = os.path.join(project_root, "logs", model_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    is_v3 = isinstance(config, TrainingConfig)

    # Display rich config panel
    _display_config(preset_name, config, env_type, n_envs, models_dir)

    # ── Build environments ───────────────────────────────────────
    if is_v3:
        env, eval_env = build_training_envs(config, env_type, n_envs, logs_dir)
    else:
        if n_envs > 1:
            env = make_vec_env(lambda: create_env(env_type), n_envs=n_envs)
        else:
            env = Monitor(create_env(env_type))
        eval_env = create_env(env_type)

    # ── Create model ─────────────────────────────────────────────
    model = _create_model(config, env, logs_dir, resume_path)

    # ── Setup callbacks ──────────────────────────────────────────
    callbacks, eval_cb = _build_callbacks(config, eval_env, models_dir, logs_dir, model_name)

    # ── Train ────────────────────────────────────────────────────
    t0 = time.time()
    console.print("[bold green]Iniciando entrenamiento…[/bold green]\n")
    model.learn(total_timesteps=config.total_timesteps, callback=callbacks, progress_bar=True)
    training_time = time.time() - t0

    # ── Save ─────────────────────────────────────────────────────
    final_path = os.path.join(models_dir, f"{model_name}_final")
    model.save(final_path)

    if is_v3 and config.use_vec_normalize:
        norm_path = os.path.join(models_dir, "vec_normalize.pkl")
        save_vec_normalize(env, norm_path)

    best_reward = eval_cb.best_mean_reward if hasattr(eval_cb, "best_mean_reward") else None
    _save_metadata(models_dir, config, env_type, n_envs,
                   best_mean_reward=best_reward, training_time=training_time)

    # Display results panel
    _display_results(model_name, models_dir, training_time, best_reward)

    env.close()
    if hasattr(eval_env, "close"):
        eval_env.close()
    return final_path


# ═══════════════════════════════════════════════════════════════════
# CLI Commands
# ═══════════════════════════════════════════════════════════════════

@app.command()
def start(
    preset: Annotated[str, typer.Option(
        "--preset", "-p",
        help="Preset de entrenamiento (ver 'presets' para lista completa)",
    )] = "v3_optimized",
    env: Annotated[str, typer.Option(
        "--env", "-e", help="Tipo de entorno: base | powerups",
    )] = "base",
    name: Annotated[Optional[str], typer.Option(
        "--name", help="Nombre personalizado del modelo",
    )] = None,
    resume: Annotated[Optional[str], typer.Option(
        "--resume", help="Ruta a modelo para reanudar entrenamiento",
    )] = None,
    n_envs: Annotated[int, typer.Option(
        "--n-envs", "-n", help="Número de entornos paralelos",
    )] = 1,
    # ── Overrides ────────────────────────────────────────────────
    timesteps: Annotated[Optional[int], typer.Option(
        "--timesteps", "-t", help="Total timesteps (override)",
    )] = None,
    epochs: Annotated[Optional[int], typer.Option(
        "--epochs", help="Epochs por update (override)",
    )] = None,
    lr: Annotated[Optional[float], typer.Option(
        "--lr", help="Learning rate fijo (override)",
    )] = None,
    batch_size: Annotated[Optional[int], typer.Option(
        "--batch-size", "-b", help="Batch size (override)",
    )] = None,
    gamma: Annotated[Optional[float], typer.Option(
        "--gamma", help="Discount factor (override)",
    )] = None,
    ent_coef: Annotated[Optional[float], typer.Option(
        "--ent-coef", help="Entropy coefficient (override)",
    )] = None,
    clip_range: Annotated[Optional[float], typer.Option(
        "--clip-range", help="PPO clip range (override)",
    )] = None,
    n_steps: Annotated[Optional[int], typer.Option(
        "--n-steps", help="Steps por rollout (override)",
    )] = None,
    checkpoint_freq: Annotated[Optional[int], typer.Option(
        "--checkpoint-freq", help="Frecuencia de checkpoint (override)",
    )] = None,
    eval_freq: Annotated[Optional[int], typer.Option(
        "--eval-freq", help="Frecuencia de evaluación (override)",
    )] = None,
):
    """Entrenar modelo con un preset y overrides opcionales."""
    if preset not in PRESETS:
        console.print(f"[red]Error: preset '{preset}' no existe.[/red]")
        console.print(f"Disponibles: {', '.join(sorted(PRESETS.keys()))}")
        raise typer.Exit(1)

    config = copy.deepcopy(PRESETS[preset])

    # Apply overrides
    _overrides: dict[str, tuple[str, object]] = {
        "total_timesteps": ("total_timesteps", timesteps),
        "n_epochs": ("n_epochs", epochs),
        "learning_rate": ("learning_rate", lr),
        "batch_size": ("batch_size", batch_size),
        "gamma": ("gamma", gamma),
        "ent_coef": ("ent_coef", ent_coef),
        "clip_range": ("clip_range", clip_range),
        "n_steps": ("n_steps", n_steps),
        "checkpoint_freq": ("checkpoint_freq", checkpoint_freq),
        "eval_freq": ("eval_freq", eval_freq),
    }
    for _, (attr, val) in _overrides.items():
        if val is not None:
            setattr(config, attr, val)

    run_training(config, env, name, resume, n_envs)


@app.command()
def wizard():
    """Asistente interactivo para configurar y lanzar entrenamiento."""
    console.print()
    console.print(Rule("[bold blue]🏒 Hockey RL — Training Wizard[/bold blue]"))
    console.print()

    # ── 1. Algorithm ─────────────────────────────────────────────
    console.print("[bold cyan]1.[/bold cyan] [bold]Algoritmo[/bold]")
    algo_t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    algo_t.add_column("#", width=3, style="dim")
    algo_t.add_column("Algo", style="bold", min_width=6)
    algo_t.add_column("Descripción")
    algo_t.add_row("1", "PPO", "Proximal Policy Optimization — estable, recomendado [green]★[/green]")
    algo_t.add_row("2", "DQN", "Deep Q-Network — off-policy, comparación")
    console.print(algo_t)
    algo_idx = IntPrompt.ask("  Seleccionar", default=1)
    algorithm = "PPO" if algo_idx != 2 else "DQN"
    console.print()

    # ── 2. Profile ───────────────────────────────────────────────
    console.print("[bold cyan]2.[/bold cyan] [bold]Perfil de entrenamiento[/bold]")

    if algorithm == "PPO":
        profiles = [
            ("v3_quick", "800K", "Pruebas rápidas (~5 min)"),
            ("v3_optimized", "2M", "Balanceado + VecNormalize [green]★[/green]"),
            ("v3_deep", "5M", "Exhaustivo (~1h)"),
            ("v3_framestack", "2M", "Temporal stacking (4 frames)"),
            ("v3_exploration", "1.5M", "Alta exploración (ent=0.03)"),
        ]
        default_prof = 2
    else:
        profiles = [
            ("v3_dqn", "2M", "DQN optimizado + VecNormalize [green]★[/green]"),
        ]
        default_prof = 1

    prof_t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    prof_t.add_column("#", width=3, style="dim")
    prof_t.add_column("Preset", style="cyan", min_width=18)
    prof_t.add_column("Steps", style="yellow", min_width=6)
    prof_t.add_column("Descripción")
    for i, (pname, steps, desc) in enumerate(profiles, 1):
        prof_t.add_row(str(i), pname, steps, desc)
    console.print(prof_t)

    prof_idx = IntPrompt.ask("  Seleccionar", default=default_prof)
    prof_idx = max(1, min(prof_idx, len(profiles)))
    preset_name = profiles[prof_idx - 1][0]
    config: TrainingConfig = copy.deepcopy(PRESETS[preset_name])  # type: ignore
    console.print()

    # ── 3. Environment ───────────────────────────────────────────
    console.print("[bold cyan]3.[/bold cyan] [bold]Entorno[/bold]")
    env_t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    env_t.add_column("#", width=3, style="dim")
    env_t.add_column("Env", style="bold", min_width=10)
    env_t.add_column("Descripción")
    env_t.add_row("1", "base", "Air Hockey estándar [green]★[/green]")
    env_t.add_row("2", "powerups", "Con power-ups (obs extendida)")
    console.print(env_t)
    env_idx = IntPrompt.ask("  Seleccionar", default=1)
    env_type = "base" if env_idx != 2 else "powerups"
    console.print()

    # ── 4. Parallel envs ─────────────────────────────────────────
    console.print("[bold cyan]4.[/bold cyan] [bold]Entornos paralelos[/bold]")
    console.print("  [dim]Más envs = más rápido, más RAM. Recomendado: 4[/dim]")
    n_envs = IntPrompt.ask("  Cantidad", default=4)
    console.print()

    # ── 5. Timesteps ─────────────────────────────────────────────
    console.print("[bold cyan]5.[/bold cyan] [bold]Total de timesteps[/bold]")
    console.print(f"  [dim]Preset sugiere {config.total_timesteps:,}[/dim]")
    ts_input = IntPrompt.ask("  Timesteps", default=config.total_timesteps)
    config.total_timesteps = ts_input
    console.print()

    # ── 6. Advanced tweaks ───────────────────────────────────────
    console.print("[bold cyan]6.[/bold cyan] [bold]Ajustes avanzados[/bold]")
    if Confirm.ask("  ¿Modificar hiperparámetros?", default=False):
        console.print()

        # LR
        lr_str = Prompt.ask(
            "  Learning rate",
            default=_lr_display(config.learning_rate),
        )
        try:
            config.learning_rate = float(lr_str)
        except ValueError:
            pass  # keep existing schedule

        config.batch_size = IntPrompt.ask("  Batch size", default=config.batch_size)
        config.n_epochs = IntPrompt.ask("  Epochs/update", default=config.n_epochs)
        config.gamma = FloatPrompt.ask("  Gamma (γ)", default=config.gamma)
        config.ent_coef = FloatPrompt.ask("  Entropy coef", default=config.ent_coef)
        config.clip_range = FloatPrompt.ask("  Clip range", default=config.clip_range)
        config.n_steps = IntPrompt.ask("  N steps (rollout)", default=config.n_steps)

        if isinstance(config, TrainingConfig):
            console.print()
            console.print("  [bold]Wrappers & observaciones:[/bold]")
            config.use_vec_normalize = Confirm.ask(
                "  VecNormalize", default=config.use_vec_normalize,
            )
            config.use_extended_obs = Confirm.ask(
                "  Extended Obs (17D)", default=config.use_extended_obs,
            )
            config.use_frame_stack = Confirm.ask(
                "  Frame Stacking", default=config.use_frame_stack,
            )
            if config.use_frame_stack:
                config.n_frame_stack = IntPrompt.ask(
                    "  N frames", default=config.n_frame_stack,
                )
    console.print()

    # ── 7. Review ────────────────────────────────────────────────
    console.print(Rule("[bold]Revisión final[/bold]"))
    model_name = f"air_hockey_{preset_name}_{env_type}"
    models_dir = os.path.join(project_root, "models", model_name)
    _display_config(preset_name, config, env_type, n_envs, models_dir)

    # ── 8. Confirm ───────────────────────────────────────────────
    if not Confirm.ask("[bold]¿Iniciar entrenamiento?[/bold]", default=True):
        console.print("[yellow]Entrenamiento cancelado.[/yellow]")
        raise typer.Exit(0)

    run_training(config, env_type, model_name, None, n_envs)


@app.command()
def presets():
    """Mostrar todos los presets disponibles con sus configuraciones."""
    table = Table(
        title="🏒 Presets de Entrenamiento Disponibles",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Preset", style="cyan bold", min_width=16)
    table.add_column("Algo", style="magenta", justify="center")
    table.add_column("Steps", style="yellow", justify="right")
    table.add_column("LR", style="green")
    table.add_column("Batch", justify="right")
    table.add_column("Net (π)", style="blue")
    table.add_column("Obs", justify="center")
    table.add_column("VecNorm", justify="center")
    table.add_column("FStack", justify="center")

    _on = "[green]✓[/green]"
    _off = "[dim]—[/dim]"

    for pname in sorted(PRESETS.keys()):
        cfg = PRESETS[pname]
        algo = getattr(cfg, "algorithm", "PPO")
        is_v3 = isinstance(cfg, TrainingConfig)
        obs = "17D" if (is_v3 and cfg.use_extended_obs) else "13D"
        vnorm = _on if (is_v3 and cfg.use_vec_normalize) else _off
        fs = f"[green]✓({cfg.n_frame_stack})[/green]" if (is_v3 and cfg.use_frame_stack) else _off

        table.add_row(
            pname, algo,
            f"{cfg.total_timesteps:,}",
            _lr_display(cfg.learning_rate),
            str(cfg.batch_size),
            str(list(cfg.net_arch_pi)),
            obs, vnorm, fs,
        )

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Uso: uv run python -m training.train start --preset <nombre>[/dim]")
    console.print()


# ═══════════════════════════════════════════════════════════════════
# Backward compatibility — used by scripts / tests
# ═══════════════════════════════════════════════════════════════════

def train(
    preset_name: str = "v3_optimized",
    env_type: str = "base",
    model_name: str | None = None,
    resume_path: str | None = None,
    n_envs: int = 1,
    args=None,
):
    """Backward-compatible train function."""
    config = PRESETS.get(preset_name)
    if config is None:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {sorted(PRESETS.keys())}")

    config = copy.deepcopy(config)

    if args is not None:
        override_map = {
            "timesteps": "total_timesteps",
            "epochs": "n_epochs",
            "lr": "learning_rate",
            "batch_size": "batch_size",
            "gamma": "gamma",
            "ent_coef": "ent_coef",
            "clip_range": "clip_range",
            "n_steps": "n_steps",
            "checkpoint_freq": "checkpoint_freq",
            "eval_freq": "eval_freq",
        }
        for cli_name, attr_name in override_map.items():
            val = getattr(args, cli_name, None)
            if val is not None:
                setattr(config, attr_name, val)

    return run_training(config, env_type, model_name, resume_path, n_envs)


if __name__ == "__main__":
    app()
