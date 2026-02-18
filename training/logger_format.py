"""Custom Rich-based terminal output format for SB3 training.

Reemplaza ``HumanOutputFormat`` con un panel compacto de altura fija que:
- Se sobreescribe in-place cada dump (cursor ANSI sube N líneas y borra)
- Muestra delta Δ (+/-) en verde/rojo cuando un valor cambia
- Ignora todos los prefijos ``rewards/`` (van solo a TensorBoard)

Uso (aplicado automáticamente en train.py):
    model.set_logger(build_rich_logger(logs_dir))
"""
from __future__ import annotations

import sys
from io import StringIO
from typing import Any, Dict, Tuple, Union

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from stable_baselines3.common.logger import KVWriter, Logger, make_output_format

# ─── Claves visibles en terminal → alias corto ───────────────────────────────
_SHOW: dict[str, str] = {
    "rollout/ep_rew_mean":           "reward",
    "rollout/ep_len_mean":           "ep_len",
    "train/explained_variance":      "expl_var",
    "train/approx_kl":               "approx_kl",
    "train/value_loss":              "val_loss",
    "train/loss":                    "loss",
    "train/entropy_loss":            "entropy",
    "train/policy_gradient_loss":    "pg_loss",
    "train/clip_fraction":           "clip_frac",
    "train/learning_rate":           "lr",
    "time/fps":                      "fps",
    "time/total_timesteps":          "steps",
    "time/iterations":               "iter",
    "time/time_elapsed":             "elapsed",
    "eval/mean_reward":              "eval_rew",
    "eval/mean_ep_length":           "eval_len",
}

# Mayor es mejor → delta verde si sube
_HIGHER_BETTER: set[str] = {
    "rollout/ep_rew_mean", "train/explained_variance",
    "eval/mean_reward", "rollout/ep_len_mean", "time/fps",
}
# Menor es mejor → delta verde si baja
_LOWER_BETTER: set[str] = {
    "train/value_loss", "train/loss",
    "train/approx_kl", "train/clip_fraction",
}

# Cambio mínimo relativo para mostrar Δ (evita ruido)
_DELTA_THRESHOLD = 0.015


def _fmt_val(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) >= 10_000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:,.1f}"
        if abs(v) >= 10:
            return f"{v:.3g}"
        return f"{v:.4g}"
    return str(v)


def _delta_text(key: str, prev: float, curr: float) -> Text:
    """Texto Rich coloreado con el delta porcentual."""
    if prev == 0:
        return Text("")
    change = (curr - prev) / (abs(prev) + 1e-9)
    if abs(change) < _DELTA_THRESHOLD:
        return Text("")

    sign = "+" if change > 0 else ""
    pct = f"{sign}{change * 100:.1f}%"

    if key in _HIGHER_BETTER:
        color = "green" if change > 0 else "red"
    elif key in _LOWER_BETTER:
        color = "green" if change < 0 else "red"
    else:
        color = "yellow"

    return Text(f" {pct}", style=f"bold {color}")


# ─────────────────────────────────────────────────────────────────────────────

class RichOutputFormat(KVWriter):
    """Panel compacto de altura fija con sobreescritura in-place y deltas Δ."""

    _UP      = "\033[{n}A"   # cursor N líneas arriba
    _CLRDOWN = "\033[J"      # borrar desde cursor hasta fin de pantalla

    def __init__(self) -> None:
        self._prev: dict[str, float] = {}
        self._last_lines: int = 0

    # ── KVWriter ─────────────────────────────────────────────────────────────

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        # Solo claves en _SHOW que no estén excluidas de stdout
        visible: dict[str, Any] = {}
        for key in _SHOW:
            if key not in key_values:
                continue
            excl = key_excluded.get(key, ())
            if isinstance(excl, str):
                excl = (excl,)
            if "stdout" in excl:
                continue
            visible[key] = key_values[key]

        if not visible:
            return

        # Renderizar a lista de líneas
        lines = self._render_lines(visible)
        output = "\n".join(lines) + "\n"

        # Sobreescritura in-place: subir cursor y borrar
        if self._last_lines > 0:
            sys.stdout.write(self._UP.format(n=self._last_lines))
            sys.stdout.write(self._CLRDOWN)

        sys.stdout.write(output)
        sys.stdout.flush()
        self._last_lines = len(lines)

        # Guardar valores actuales como referencia para el próximo delta
        for key, val in visible.items():
            if isinstance(val, (int, float)):
                self._prev[key] = float(val)

    def close(self) -> None:
        pass

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render_lines(self, visible: dict[str, Any]) -> list[str]:
        buf = StringIO()
        try:
            width = Console().width or 100
        except Exception:
            width = 100
        cap = Console(file=buf, highlight=False, markup=True,
                      width=width, no_color=False)
        self._render(cap, visible)
        return buf.getvalue().rstrip("\n").split("\n")

    def _render(self, console: Console, visible: dict[str, Any]) -> None:
        # ── Cabecera dinámica ────────────────────────────────────────
        td = {_SHOW[k]: visible[k] for k in visible if k in _SHOW}
        steps   = f"[green bold]{int(td.get('steps', 0)):,}[/green bold]"
        fps_val = td.get('fps', '?')
        fps     = f"[dim]{fps_val} fps[/dim]"
        it      = f"iter [bold]{td.get('iter', '?')}[/bold]"
        elapsed = f"[dim]{td.get('elapsed', '?')}s[/dim]"
        console.rule(
            f"🏒 PPO  {steps} steps  {fps}  {it}  {elapsed}",
            style="blue",
        )

        # ── Agrupar por sección ──────────────────────────────────────
        _COLORS = {"rollout": "bold green", "eval": "bold yellow", "train": "bold cyan"}
        sections: dict[str, list[tuple[str, Any, Text]]] = {}

        for key, val in visible.items():
            section = key.split("/")[0]
            if section == "time":
                continue
            alias = _SHOW.get(key, key.split("/")[-1])
            prev = self._prev.get(key)
            delta = (
                _delta_text(key, prev, float(val))
                if prev is not None and isinstance(val, (int, float))
                else Text("")
            )
            sections.setdefault(section, []).append((alias, val, delta))

        # ── Una tabla por sección ────────────────────────────────────
        for section in ("rollout", "eval", "train"):
            if section not in sections:
                continue
            color = _COLORS.get(section, "white")
            console.print(f"  [{color}]{section.upper()}[/{color}]")

            tbl = Table(box=box.SIMPLE, show_header=False,
                        padding=(0, 1), expand=True)
            # 2 pares (alias | valor+delta) por fila → 4 columnas
            tbl.add_column(ratio=2)  # alias1
            tbl.add_column(ratio=3)  # val1
            tbl.add_column(ratio=2)  # alias2
            tbl.add_column(ratio=3)  # val2

            row: list[Any] = []
            for alias, val, delta in sections[section]:
                val_text = Text(_fmt_val(val), style="bold white")
                val_text.append_text(delta)
                row.append(Text(alias, style="dim"))
                row.append(val_text)
                if len(row) == 4:
                    tbl.add_row(*row)
                    row = []
            if row:
                while len(row) < 4:
                    row.append(Text(""))
                tbl.add_row(*row)

            console.print(tbl)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_rich_logger(logs_dir: str) -> Logger:
    """Logger SB3: terminal Rich (in-place + deltas) + TensorBoard + CSV."""
    return Logger(
        folder=logs_dir,
        output_formats=[
            RichOutputFormat(),
            make_output_format("tensorboard", logs_dir),
            make_output_format("csv", logs_dir),
        ],
    )
