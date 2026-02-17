# Rúbrica de Rewards v2 — Air Hockey RL Agent

> **Fecha:** 16 de febrero de 2026  
> **Versión:** 2.0  
> **Basado en:** Mejora 7 del plan de implementación  
> **Fundamentación:** Ng et al. (1999) PBRS, Bansal et al. (2018) self-play, SB3 best practices

---

## Arquitectura del Sistema de Rewards

El sistema usa una **arquitectura componentizada** donde cada tipo de reward es un módulo independiente orquestado por `RewardCalculator`. Esto permite:

- **Extensibilidad:** Agregar/quitar componentes sin modificar código existente
- **Trazabilidad:** Cada contribución se registra en `RewardBreakdown`
- **Ablación:** `RewardCalculator.minimal()` para comparaciones experimentales
- **Mantenibilidad:** SRP estricto — cada componente tiene una sola responsabilidad

```
RewardCalculator (orquestador)
├── GoalRewardComponent          [sparse, eventos]
├── HitRewardComponent           [eventos, anti-exploit]
├── ShotDirectionComponent       [eventos, NUEVO]
├── ClearRewardComponent         [eventos, NUEVO]
├── InterceptionComponent        [eventos, defensivo]
├── CounterattackComponent       [eventos, transición]
├── PositionalRewardComponent    [denso, posicional]
├── DefensiveRewardComponent     [denso, NUEVO]
├── GoalPressureComponent        [denso, NUEVO]
├── DisciplineComponent          [denso, penalizaciones]
└── PBRSComponent                [denso, potential-based]
```

---

## Rúbrica Completa

### Rewards Ofensivos

| Componente | Valor | Condición | Justificación |
|---|---|---|---|
| **Hit base** | +0.8 × discount | Golpear el puck. Discount = 0.8^n por hits consecutivos (anti-exploit) | Incentivo base para contacto con el puck |
| **Shot Quality** | +2.5 × alignment × speed | `alignment` = dot(puck_dir, to_goal) > 0, `speed` = puck_speed/max_speed | Recompensa tiros dirigidos a portería con velocidad |
| **Hard Shot** | +0.5 | speed_ratio > 0.6 | Bonus por tiros potentes |
| **Shot to Goal** *(NUEVO)* | +1.5 × dot_product | dot(puck_vel, direction_to_goal) > 0.7 post-golpe | Recompensa precisa por dirección hacia portería |
| **Shot Away** *(NUEVO)* | -0.5 | dot(puck_vel, direction_to_goal) < 0 post-golpe | Penaliza golpes en dirección incorrecta |
| **Goal Scored** | +5.0 | AI anota gol | Señal principal ofensiva (sparse) |
| **Goal Pressure** *(NUEVO)* | +0.03/frame | Puck en mitad rival y yendo hacia portería (vx < -0.5) | Incentiva mantener presión ofensiva |
| **Counterattack** | +0.5 | Hit defensivo previo → puck ahora va hacia portería rival | Transición defensa→ataque |

### Rewards Defensivos

| Componente | Valor | Condición | Justificación |
|---|---|---|---|
| **Interception** | +1.0 | Hit puck que venía hacia portería AI (vx > 2.0) | Paradas defensivas |
| **Clear** *(NUEVO)* | +1.5 | Puck venía (vx > 1) pre-hit, sale (vx < -1) post-hit | Despejes exitosos |
| **Block Position** *(NUEVO)* | +0.1 × y_coverage | Entre puck y portería, puck en AI half + heading toward AI | Posicionamiento de bloqueo |
| **Def. Proximity** *(NUEVO)* | +0.05 × prox | < 100px del puck cuando amenaza (heading toward AI) | Cercanía defensiva reactiva |
| **Goal Conceded** | -4.0 | Rival anota (antes -3.0) | Penalización incrementada |
| **Negligence** *(NUEVO)* | -1.0 | Conceder gol estando lejos de portería propia (>60% field) | Penaliza abandono posicional |

### Rewards Posicionales

| Componente | Valor | Condición | Justificación |
|---|---|---|---|
| **Gap Control** | +0.05 max | ~100px del puck cuando en AI half | Distancia óptima de reacción |
| **Approach** | +0.08 × factor | Reducir distancia cuando > 150px | Incentiva acercarse |
| **Positional Play** | +0.03 × coverage | Entre puck y portería propia | Cobertura defensiva |
| **Y-Alignment** | +0.02 × alignment | Alineación vertical con puck | Cobertura de ángulo |
| **Pressure Play** | +0.01 × proximity | Cerca del centro con puck en mitad rival | Presión posicional |

### Penalizaciones (Disciplina)

| Componente | Valor | Condición | Justificación |
|---|---|---|---|
| **Net-Front** | -0.05 | Demasiado cerca de portería propia (< 2× radius) | Evita camping |
| **Inactivity** | -0.02 | Quieto con puck cerca (< 200px) en AI half | Evita pasividad |

### PBRS (Potential-Based Reward Shaping)

| Componente | Fórmula | Justificación |
|---|---|---|
| **Shaping** | F = γ·Φ(s') − Φ(s) | Preserva política óptima (Ng et al., 1999) |

Función de potencial:
$$\Phi(s) = 0.5 \times (1 - \frac{puck_x}{W}) + 0.2 \times defense\_alignment$$

---

## Cambios respecto a v1

| Aspecto | v1 | v2 |
|---|---|---|
| Arquitectura | Método monolítico `_calculate_reward()` | Componentes independientes + orquestador |
| Gol concedido | -3.0 | -4.0 + negligencia (-1.0) |
| Shot direction | Solo `shot_quality` básico | Shot to goal (+1.5×dp) y shot away (-0.5) |
| Despejes | No existía | Clear (+1.5) por inversión de dirección |
| Defensa posicional | No existía | Block position (+0.1) y proximity (+0.05) |
| Presión ofensiva | No existía | Goal pressure (+0.03/frame) |
| Anti-exploit | No existía | Diminishing returns en hits consecutivos |
| PBRS | No existía | Γ·Φ(s')−Φ(s) con potencial basado en progreso |
| Trazabilidad | Ninguna | `RewardBreakdown` con desglose por componente |
| Extensibilidad | Requiere modificar método | Agregar clase → registrar en `default()` |

---

## Medidas Anti-Exploit

1. **Hit farming prevention:** Discount 0.8^n por hits consecutivos (base 0.8 → ~0.26 después de 5)
2. **Defensive bias prevention:** Ratio ofensivo:defensivo >2:1 en magnitud máxima
3. **PBRS correctness:** Función de potencial solo depende del estado, usa γ correcto
4. **Penalización por negligencia:** Evita que el agente ignore la defensa
5. **Shot direction penalty:** Evita golpes sin propósito (hit farming sin dirección)

---

## Métricas de Monitoreo Recomendadas

Al evaluar los modelos, monitorizar:

- **Goles anotados / episodio** (no solo reward acumulado)
- **Ratio goles a favor / en contra**
- **Desglose de reward por categoría** (usar `breakdown.categories`)
- **Distribución de velocidad del puck al contacto**
- **Steps promedio entre hits**
- **% del tiempo en posición defensiva correcta** (block position activo)

---

## Uso en Código

```python
from training.envs.rewards import RewardCalculator, FieldState

# Configuración por defecto (todos los componentes)
calc = RewardCalculator.default(gamma=0.995)

# Solo sparse (para ablación)
calc = RewardCalculator.minimal()

# En cada step:
reward, breakdown = calc.calculate(field_state)
print(breakdown.components)   # {'hit_base': 0.8, 'shot_quality': 1.2, ...}
print(breakdown.categories)   # {RewardCategory.HIT: 2.0, ...}
print(breakdown.total)         # 2.5
```

---

## Referencias

- Ng, A., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations*. ICML.
- Bansal, T. et al. (2018). *Emergent Complexity via Multi-Agent Competition*. ICLR.
- Skalse, J. et al. (2022). *Defining and Characterizing Reward Hacking*. NeurIPS.
- van Seijen, H. et al. (2017). *Hybrid Reward Architecture for RL*. arXiv:1706.04208.
- Stable-Baselines3 documentation: Custom environments & reward design.
