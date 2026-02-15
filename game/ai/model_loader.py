"""
Model loading and management for RL agents.
Finds the best available model by reading evaluation metadata,
then falls back to most-recently modified .zip file.
"""
import os
import json
import glob
import torch
import numpy as np


MODEL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
)


def find_best_model(project_root: str = None) -> tuple:
    """
    Find the best available trained model.

    Strategy:
      1. Scan all models/<run>/best_model/best_model.zip that have a
         companion metadata.json — pick the one with the highest
         'best_mean_reward'.
      2. If tied or no metadata, prefer the most recently modified file.
      3. Fall back to any *_final.zip found under run dirs.

    Returns (model_path, model_type) or (None, None).
    """
    models_dir = MODEL_ROOT
    if project_root:
        models_dir = os.path.join(project_root, "models")

    if not os.path.isdir(models_dir):
        return None, None

    candidates: list[dict] = []  # {path, score, mtime}

    # ---- 1. best_model directories with metadata ----
    for run_dir in _iter_run_dirs(models_dir):
        best_zip = os.path.join(run_dir, "best_model", "best_model.zip")
        if not os.path.isfile(best_zip):
            continue
        meta = _load_metadata(run_dir)
        score = meta.get("best_mean_reward", None) if meta else None
        mtime = os.path.getmtime(best_zip)
        candidates.append({"path": best_zip, "score": score, "mtime": mtime})

    # ---- 2. *_final.zip in run dirs ----
    for run_dir in _iter_run_dirs(models_dir):
        for fzip in glob.glob(os.path.join(run_dir, "*_final.zip")):
            meta = _load_metadata(run_dir)
            score = meta.get("best_mean_reward", None) if meta else None
            mtime = os.path.getmtime(fzip)
            candidates.append({"path": fzip, "score": score, "mtime": mtime})

    # ---- 3. Any .zip directly under models/ (legacy) ----
    for fzip in glob.glob(os.path.join(models_dir, "*.zip")):
        mtime = os.path.getmtime(fzip)
        candidates.append({"path": fzip, "score": None, "mtime": mtime})

    if not candidates:
        return None, None

    # Sort: highest score first (None goes last), then most recent
    def _sort_key(c):
        s = c["score"] if c["score"] is not None else float("-inf")
        return (s, c["mtime"])

    candidates.sort(key=_sort_key, reverse=True)
    best = candidates[0]
    print(f"[ModelLoader] Selected: {best['path']}")
    if best["score"] is not None:
        print(f"[ModelLoader]   score={best['score']:.2f}")
    return best["path"], "auto"


def _iter_run_dirs(models_dir: str):
    """Yield subdirectories of models_dir (each is a training run)."""
    if not os.path.isdir(models_dir):
        return
    for name in sorted(os.listdir(models_dir)):
        p = os.path.join(models_dir, name)
        if os.path.isdir(p):
            yield p


def _load_metadata(run_dir: str) -> dict | None:
    """Load metadata.json from a run directory."""
    meta_path = os.path.join(run_dir, "metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            pass
    # Try reading evaluations.npz from the run dir or logs
    for candidate_path in [
        os.path.join(run_dir, "evaluations.npz"),
        os.path.join(
            os.path.dirname(run_dir), "..", "logs",
            os.path.basename(run_dir), "evaluations.npz",
        ),
    ]:
        if os.path.isfile(candidate_path):
            try:
                data = np.load(candidate_path)
                results = data["results"]  # shape (n_evals, n_eval_episodes)
                mean_rewards = results.mean(axis=1)
                return {"best_mean_reward": float(mean_rewards.max())}
            except Exception:
                pass
    return None


def load_optimized_model(model_path: str, model_type: str = "auto"):
    """
    Load and optimize an RL model for inference.
    Returns (model, detected_model_type).
    """
    model = None

    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
    except Exception:
        try:
            from stable_baselines3 import DQN
            model = DQN.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Could not load model from {model_path}: {e}")

    # Auto-detect type from observation space
    obs_space = model.observation_space
    if hasattr(obs_space, "shape"):
        obs_dim = obs_space.shape[0]
        if obs_dim == 26:
            model_type = "v2_powerups"
        elif obs_dim == 21:
            model_type = "enhanced"
        elif obs_dim == 13:
            # Check action space to distinguish v1 vs v2
            act_n = model.action_space.n if hasattr(model.action_space, "n") else 5
            model_type = "v2" if act_n == 9 else "original"
        else:
            model_type = "unknown"

    # Optimize for inference
    model.policy.set_training_mode(False)
    if torch.cuda.is_available():
        model.policy = model.policy.to("cuda")
    else:
        torch.set_num_threads(4)

    print(f"[ModelLoader] Loaded {model_type} model "
          f"(obs={obs_space.shape}, actions={model.action_space.n})")
    return model, model_type
