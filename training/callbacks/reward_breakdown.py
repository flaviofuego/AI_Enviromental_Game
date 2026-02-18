"""Callback que registra el desglose de rewards por componente en TensorBoard.

Estrategia:
- Cada step, lee `_last_reward_breakdown` de todos los envs del VecEnv.
- Acumula los valores por componente durante el episodio.
- Al terminar cada episodio (done=True), guarda el acumulado.
- Cada `log_freq` timesteps loguea el promedio de episodios completados
  a TensorBoard bajo el prefijo `rewards/`.

Sin output a terminal (verbose=0 por defecto).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class RewardBreakdownCallback(BaseCallback):
    """Registra rewards por componente en TensorBoard sin prints a terminal.

    Cada componente del sistema de rewards (``hit_base``, ``shot_quality``,
    ``goal_scored``, etc.) aparecerá como una métrica separada en TensorBoard
    bajo la sección ``rewards/``.

    Parameters
    ----------
    log_freq : int
        Cada cuántos timesteps se vuelca el promedio a TensorBoard.
        Defecto: 4096 (equivale a un rollout con n_steps=2048 y 2 envs).
    verbose : int
        0 = silencioso (recomendado), 1 = resumen breve a terminal.
    """

    def __init__(self, log_freq: int = 4096, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._log_freq = log_freq

        # Acumuladores por env (se inicializan en _on_training_start)
        self._episode_sums: list[dict[str, float]] = []

        # Buffer de episodios completos pendientes de loguear
        self._pending: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._episode_sums = [defaultdict(float) for _ in range(n)]

    def _on_step(self) -> bool:
        # ── Obtener breakdowns del VecEnv ────────────────────────────
        try:
            breakdowns = self.training_env.get_attr("_last_reward_breakdown")
        except Exception:
            # Entorno no soporta el atributo → no hacer nada
            return True

        dones: list[bool] = list(self.locals.get("dones", []))
        if not dones:
            dones = [False] * len(breakdowns)

        # ── Acumular y detectar fin de episodio ──────────────────────
        for i, bd in enumerate(breakdowns):
            if bd is None:
                continue
            for comp, val in bd.components.items():
                self._episode_sums[i][comp] += val

            if i < len(dones) and dones[i]:
                # Episodio terminado → guardar copia y resetear
                self._pending.append(dict(self._episode_sums[i]))
                self._episode_sums[i] = defaultdict(float)

        # ── Volcar a TensorBoard cada log_freq ───────────────────────
        if self.num_timesteps % self._log_freq < self.training_env.num_envs and self._pending:
            self._flush_to_tensorboard()

        return True

    def _on_training_end(self) -> None:
        """Volcar cualquier dato pendiente al terminar."""
        if self._pending:
            self._flush_to_tensorboard()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_to_tensorboard(self) -> None:
        """Promediar episodios pendientes y registrarlos en el logger."""
        if not self._pending:
            return

        # Unión de todos los componentes vistos
        all_keys: set[str] = set()
        for ep in self._pending:
            all_keys.update(ep.keys())

        # exclude="stdout" → solo van a TensorBoard/CSV, nunca a terminal
        _excl = ("stdout", "log")
        for key in sorted(all_keys):
            values = [ep[key] for ep in self._pending if key in ep]
            if values:
                mean_val = sum(values) / len(values)
                self.logger.record(f"rewards/{key}", mean_val, exclude=_excl)

        n_eps = len(self._pending)
        self.logger.record("rewards/_episodes_logged", n_eps, exclude=_excl)

        if self.verbose >= 1:
            top = sorted(
                {k: sum(ep.get(k, 0) for ep in self._pending) / n_eps
                 for k in all_keys}.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
            summary = "  ".join(f"{k}={v:+.3f}" for k, v in top)
            print(f"[RewardBreakdown] step={self.num_timesteps}  top5: {summary}")

        self._pending.clear()
