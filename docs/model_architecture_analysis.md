# Análisis de Arquitectura del Agente RL — v3

> **Fecha:** 16 de febrero de 2026  
> **Versión:** 3.0  
> **Estado:** Configuraciones implementadas, pendiente de entrenamiento comparativo

---

## Resumen ejecutivo

Este documento describe las optimizaciones de arquitectura del agente RL implementadas
en la versión 3 del sistema de entrenamiento. Se añaden nuevos presets, schedules de
learning rate, observaciones extendidas, normalización de entorno y soporte para DQN
como alternativa a PPO.

---

## 1. Schedules de Learning Rate

### Módulo: `training/configs/schedules.py`

Se creó un módulo reutilizable con 6 tipos de schedule:

| Schedule | Fórmula | Mejor para |
|----------|---------|------------|
| `linear_schedule` | `f + (i-f) × progress` | Decaimiento uniforme simple |
| `cosine_schedule` | Cosine annealing | Decaimiento suave, fine-tuning final |
| `cosine_warmup_schedule` | Warmup lineal + cosine decay | **Recomendado** — estabilidad + exploración |
| `constant_schedule` | Valor fijo | Baselines / ablation tests |
| `stepped_schedule` | Drops discretos | LR drops a mitad de training |
| `exponential_schedule` | Decay exponencial | Exploración agresiva → fine-tuning |

### Justificación: Warmup + Cosine

Es el schedule recomendado para PPO en juegos adversariales:

1. **Warmup (10% inicial):** Permite que los pesos se estabilicen antes de aplicar
   learning rate alto. Evita actualizaciones destructivas en las primeras iteraciones
   cuando advantage estimates son ruidosos.
2. **Cosine decay:** Más suave que linear — decae lento al inicio (exploración
   extendida), luego más rápido en el medio, y gentilmente se acerca al mínimo
   en las últimas épocas (consolidación de política).

---

## 2. Espacio de Observación

### Módulo: `training/envs/observation_builder.py`

Se implementó un `ObservationBuilder` composable con 13 features atómicas:

| Feature | Dims | Descripción |
|---------|------|-------------|
| `AI_POS` | 2 | Posición del mallet IA (x/W, y/H) |
| `PUCK_POS` | 2 | Posición del puck |
| `PUCK_VEL` | 2 | Velocidad del puck (normalizada) |
| `DISTANCE` | 1 | Distancia AI-puck / diagonal |
| `PUCK_TO_GOAL` | 1 | Distancia puck a portería enemiga |
| `PUCK_PROGRESS` | 1 | Progreso X del puck |
| `STEPS_SINCE_HIT` | 1 | Pasos desde último golpe |
| `HEADING_FLAG` | 1 | ¿Puck va hacia IA? |
| `SCORES` | 2 | Marcador normalizado |
| `OPPONENT_POS` | 2 | **NUEVO** — Posición del oponente |
| `OPPONENT_VEL` | 2 | **NUEVO** — Velocidad del oponente |
| `AI_VEL` | 2 | **NUEVO** — Velocidad propia del mallet IA |
| `GOAL_DISTANCES` | 2 | **NUEVO** — Distancia a ambas porterías |

### Presets de observación

| Preset | Dims | Features |
|--------|------|----------|
| `standard()` | 13D | v2 compatible — sin info del oponente |
| `extended()` | 17D | + opponent_pos + opponent_vel |
| `full()` | 21D | + ai_vel + goal_distances |

### Ventajas del diseño composable

- **Extensibilidad:** Agregar una nueva feature es agregar un enum + bounds + builder case.
- **Backward compatibility:** `standard()` produce exactamente los mismos 13 valores que v2.
- **Debuggability:** `builder.describe()` retorna nombres legibles para logging.
- **Experimentación:** Crear combinaciones custom para ablation studies.

---

## 3. Presets v3

### Módulo: `training/configs/v3_configs.py`

Se creó `TrainingConfig` que extiende `PPOConfig` con:

- Selección de algoritmo (PPO / DQN)
- Flags de wrappers (VecNormalize, VecFrameStack)
- Flag de observación extendida
- Ortho init y normalize_advantage
- Parámetros DQN-específicos

### Tabla comparativa de presets

| Preset | Algo | Steps | LR Schedule | Arch (pi/vf) | Obs | VecNorm | FrameStack | Justificación |
|--------|------|-------|-------------|---------------|-----|---------|------------|---------------|
| `v3_quick` | PPO | 800K | warmup+cosine | [256,128]/[256,128] | 13D | ✗ | ✗ | Baseline rápido v3 |
| `v3_optimized` | PPO | 2M | warmup+cosine | [256,256]/[128,128] | 17D | ✓ | ✗ | **Recomendado** — balance rendimiento/costo |
| `v3_deep` | PPO | 5M | warmup+cosine | [512,256,128]/[256,128] | 17D | ✓ | ✗ | Entrenamiento largo para skill máximo |
| `v3_framestack` | PPO | 2M | warmup+cosine | [256,256]/[128,128] | 17D×4=68D | ✓ | ✓(4) | Contexto temporal explícito |
| `v3_dqn` | DQN | 2M | cosine | [256,256] | 17D | ✓ | ✗ | Comparación PPO vs DQN |
| `v3_exploration` | PPO | 1.5M | warmup+cosine | [256,256,128]/[256,128] | 13D | ✗ | ✗ | Alta entropía para evitar convergencia prematura |

### Justificación de hiperparámetros clave (v3_optimized)

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| LR | warmup 10% → 3e-4 → 1e-6 | Warmup estabiliza pesos; cosine suaviza convergencia |
| n_steps | 4096 | Rollouts largos capturan dinámicas adversariales (rallies completos) |
| batch_size | 256 | Balance entre varianza del gradiente y eficiencia computacional |
| gamma | 0.997 | High discount para valorar goles distantes (episodios ~2000 steps) |
| gae_lambda | 0.98 | Alto para reducir bias en advantage estimation en juegos con rewards espaciados |
| clip_range | 0.2 | Standard; no demasiado restrictivo para permitir actualizaciones sustanciales |
| ent_coef | 0.02 | Más alto que v2 (0.015) — más exploración de las 9 acciones |
| target_kl | 0.025 | Early stopping por KL para estabilidad sin ser demasiado conservador |
| pi: [256,256] | Ancho > profundo | Para Discrete(9) con 17 obs, 2 capas anchas capturan políticas complejas sin overfitting |
| vf: [128,128] | Más estrecho que pi | Value function necesita menos capacidad que la política — reduce parámetros en ~60% vs v2 |
| ortho_init | True | Inicialización ortogonal mejora flujo de gradiente en redes profundas (estándar en PPO) |
| normalize_advantage | True | Normaliza advantage estimates por batch — crítico en envs con rewards de escala variable |
| VecNormalize | True | Normaliza obs/rewards online → rango estable para la red. Crítico con obs extendido |

---

## 4. Pipeline de Entorno

### Módulo: `training/env_pipeline.py`

Se creó una pipeline composable para wrapping de entornos:

```
Raw Env → DummyVecEnv/make_vec_env → VecNormalize (opcional) → VecFrameStack (opcional)
```

Funciones:
- `build_training_envs(config, env_type, n_envs)` — construye train + eval envs
- `save_vec_normalize(env, path)` — guarda estadísticas de normalización
- `sync_vec_normalize(train, eval)` — sincroniza stats entre train/eval

### VecNormalize

Normaliza observaciones y rewards online usando running mean/std:

- **Obs:** Centra en mean=0, std=1 → red recibe inputs en rango estable
- **Rewards:** Normaliza por running std → escala consistent sin importar reward shaping
- **clip_obs=10, clip_reward=10** → protege contra outliers

### VecFrameStack

Apila N frames consecutivos como una sola observación:

- 17D × 4 frames = 68D observation
- Da contexto temporal implícito: el agente puede "ver" velocidad y aceleración
- Alternativa a redes recurrentes (LSTM) sin la complejidad adicional

---

## 5. Soporte DQN

Se agregó soporte para DQN como alternativa a PPO:

| Aspecto | PPO | DQN |
|---------|-----|-----|
| Tipo | On-policy | Off-policy |
| Sample efficiency | Menor | Mayor (replay buffer) |
| Estabilidad | Alta | Media |
| Actions | Discrete y Continuous | Solo Discrete |
| Memoria | Baja | Alta (buffer de 200K) |

Para nuestro Discrete(9), DQN con Double Q-Learning puede ser más sample-efficient.
El preset `v3_dqn` está configurado para comparación directa.

---

## 6. Sistema de Comparación

### Módulo: `training/analysis/model_comparison.py`

Se refactorizó el comparador con:

- **`ModelEvaluator`**: Clase que evalúa modelos con métricas granulares
- **`ModelResult`**: Dataclass con win_rate, goals, episode_length, metadata
- **`print_comparison()`**: Tabla formateada con ranking
- **`results_to_dict()`**: Exportación a JSON

Métricas recopiladas por modelo:
- Mean/Std reward
- Win rate (% de episodios ganados)
- Goals scored / conceded / differential
- Average episode length

---

## 7. Protocolo de Experimentación Recomendado

### Fase 1: Quick Comparison (≈2h total)

```bash
# Baseline v2
uv run python -m training.train --preset v2_quick --env base --n-envs 4

# v3 quick (same budget, better schedule)
uv run python -m training.train --preset v3_quick --env base --n-envs 4

# v3 optimized (extended obs + normalization)
uv run python -m training.train --preset v3_optimized --env base --n-envs 4 --timesteps 800000
```

### Fase 2: Extended Training (≈6h total)

```bash
# v3 optimized full
uv run python -m training.train --preset v3_optimized --env base --n-envs 4

# v3 DQN comparison
uv run python -m training.train --preset v3_dqn --env base

# v3 framestack
uv run python -m training.train --preset v3_framestack --env base --n-envs 4
```

### Fase 3: Analysis

```python
from training.analysis.model_comparison import ModelEvaluator
from training.envs.base_env import AirHockeyEnv

evaluator = ModelEvaluator(env_factory=lambda: AirHockeyEnv(), n_eval_episodes=50)
results = evaluator.evaluate_all([
    "models/air_hockey_v2_quick_base/best_model/best_model",
    "models/air_hockey_v3_quick_base/best_model/best_model",
    "models/air_hockey_v3_optimized_base/best_model/best_model",
    "models/air_hockey_v3_dqn_base/best_model/best_model",
])
evaluator.print_comparison(results)
```

---

## 8. Archivos Creados / Modificados

### Creados
| Archivo | Propósito |
|---------|----------|
| `training/configs/schedules.py` | Schedules de LR reutilizables |
| `training/configs/v3_configs.py` | Presets v3 con `TrainingConfig` |
| `training/envs/observation_builder.py` | Observaciones composables |
| `training/env_pipeline.py` | Pipeline de wrapping de entornos |
| `docs/model_architecture_analysis.md` | Este documento |

### Modificados
| Archivo | Cambio |
|---------|--------|
| `training/train.py` | Soporte v3 + DQN + env pipeline |
| `training/envs/base_env.py` | ObservationBuilder inyectable |
| `training/envs/__init__.py` | Exports actualizados |
| `training/configs/__init__.py` | Exports de schedules + v3 |
| `training/analysis/__init__.py` | Exports de ModelEvaluator |
| `training/analysis/model_comparison.py` | Evaluador rico con métricas granulares |

---

## 9. Criterios de Éxito

| Criterio | Métrica | Target |
|----------|---------|--------|
| Superar v2_quick | mean_reward | > 209.14 |
| Win rate competitivo | win_rate | > 60% |
| Goal differential positivo | GD | > +1.0 |
| Training estable | std_reward | < 50% del mean |
| Al menos 3 variantes evaluadas | count | ≥ 3 |
| Documentación completa | este doc | ✓ |
