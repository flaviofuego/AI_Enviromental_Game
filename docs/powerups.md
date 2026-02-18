# Plan de Implementación: Sistema de Powerups Modular

> **Estado:** Planificación  
> **Fecha de creación:** 17 de febrero de 2026  
> **Versión objetivo del juego:** v2.0 (Powerups Modulares)

---

## Índice

1. [Sistema Actual de Powerups (Pre-migración)](#1-sistema-actual-de-powerups-pre-migración)
2. [Arquitectura del Nuevo Sistema](#2-arquitectura-del-nuevo-sistema)
3. [Catálogo de Powerups por Fase](#3-catálogo-de-powerups-por-fase)
4. [Sistema de Representación Visual](#4-sistema-de-representación-visual)
5. [Sistema de Notificaciones](#5-sistema-de-notificaciones)
6. [Integración con HUD](#6-integración-con-hud)
7. [Integración con Entornos de Entrenamiento](#7-integración-con-entornos-de-entrenamiento)
8. [Integración con el Botón de Ayuda](#8-integración-con-el-botón-de-ayuda)
9. [Plan de Fases e Implementación por Tarea](#9-plan-de-fases-e-implementación-por-tarea)
10. [Asignación de Responsabilidades por Subagente/Skill](#10-asignación-de-responsabilidades-por-subagenteskill)
11. [Criterios de Aceptación y Tests](#11-criterios-de-aceptación-y-tests)
12. [Configuración y Variables de Ajuste](#12-configuración-y-variables-de-ajuste)

---

## 1. Sistema Actual de Powerups (Pre-migración)

> **⚠️ Con la implementación del nuevo sistema, este código será eliminado.**  
> Esta sección documenta el estado completo del sistema existente antes de su remoción.

### 1.1 Archivos afectados

| Archivo | Rol actual | Acción |
|---|---|---|
| `game/entities/powerups.py` | Lógica central: `PowerUpType`, `PowerUpEffect`, `PowerUpConfig`, `PowerUp`, `PowerUpManager` | **Eliminar y reemplazar** |
| `training/envs/powerups_env.py` | Entorno de entrenamiento con powerups duplicado | **Eliminar y reemplazar** |
| `game/core/game_engine.py` | Referencia a `powerup_manager` | **Refactorizar** |
| `game/ui/hud.py` | `_draw_powerup_indicators()` | **Refactorizar** |

### 1.2 Powerups actuales en detalle

#### `SPEED_BOOST` — "Viento Limpio"
- **Función:** Aumenta la velocidad del mazo del jugador en un `+50%`.
- **Duración:** 5.0 segundos.
- **Objetivo:** El propio jugador (`affects_self = True`).
- **Multiplicador:** `1.5` sobre `speed_multiplier`.
- **Color:** Cyan claro `(100, 220, 255)` — Ícono: `"W"`.
- **Efectos secundarios:** Ninguno. Al expirar, `speed_multiplier` vuelve a `1.0`.
- **Método de remoción:** `players[idx].speed_multiplier = 1.0`.

#### `SIZE_INCREASE` — "Crecimiento"
- **Función:** Aumenta el tamaño del mazo del jugador en un `+30%`.
- **Duración:** 7.0 segundos.
- **Objetivo:** El propio jugador (`affects_self = True`).
- **Multiplicador:** `1.3` sobre `apply_size_modifier()`.
- **Color:** Verde `(80, 200, 80)` — Ícono: `"T"`.
- **Efectos secundarios:** Cambia hitbox y sprite del mazo. Al expirar, se llama `apply_size_modifier(1.0)`.

#### `SLOW_OPPONENT` — "Contaminación"
- **Función:** Reduce la velocidad del mazo del oponente en `-30%`.
- **Duración:** 5.0 segundos.
- **Objetivo:** El oponente (`affects_self = False`).
- **Multiplicador:** `0.7` sobre `speed_multiplier` del oponente.
- **Color:** Marrón `(150, 100, 50)` — Ícono: `"P"`.
- **Efectos secundarios:** Ninguno. Al expirar, `speed_multiplier` del oponente vuelve a `1.0`.

#### `PUCK_MAGNET` — "Campo Magnético"
- **Función:** Aplica una fuerza de atracción constante sobre el puck hacia el jugador.
- **Duración:** 4.0 segundos.
- **Objetivo:** El propio jugador (`affects_self = True`).
- **Multiplicador:** Fuerza de atracción constante de `0.3` velocidad/frame.
- **Color:** Morado `(200, 50, 200)` — Ícono: `"M"`.
- **Efectos secundarios:** Puede generar movimiento indefinido del puck si no se parchea el radio mínimo de `10px`.
- **Lógica especial:** Se aplica en cada `update()` del `PowerUpManager`, no en `_apply_powerup`.

#### `SHIELD` — "Barrera de Coral"
- **Función:** Activa un escudo que protege la portería del jugador de goles.
- **Duración:** 3.0 segundos.
- **Objetivo:** El propio jugador (`affects_self = True`).
- **Color:** Naranja-coral `(255, 150, 100)` — Ícono: `"S"`.
- **Efectos secundarios:** Modifica `shield_active[collector_idx] = True` en `PowerUpManager`. La lógica de intercepción real del gol **NO está implementada** en `game_engine.py` (bug conocido).
- **Nota:** La detección de colisión con el escudo debe ser implementada por el equipo.

#### `SHRINK_OPPONENT` — "Sequía"
- **Función:** Reduce el tamaño del mazo del oponente en `-25%`.
- **Duración:** 5.0 segundos.
- **Objetivo:** El oponente (`affects_self = False`).
- **Multiplicador:** `0.75` sobre `apply_size_modifier()` del oponente.
- **Color:** Dorado `(220, 180, 50)` — Ícono: `"D"`.
- **Efectos secundarios:** Si el oponente también tiene `SIZE_INCREASE`, los efectos se solapan con multiplicadores independientes (bug conocido: no hay composición de modificadores).

### 1.3 Problemas críticos del sistema actual

| Problema | Descripción |
|---|---|
| **Duplicación de código** | `PowerUpType` y la lógica de efectos están duplicados entre `game/entities/powerups.py` y `training/envs/powerups_env.py` |
| **Hardcoding del motor** | Los casos `if pu.type == PowerUpType.X` en `_apply_powerup` y `_remove_effect` requieren modificar `PowerUpManager` para cada nuevo powerup |
| **Sin composición** | Si dos powerups afectan el mismo atributo (`speed_multiplier`), el segundo sobreescribe al primero en lugar de acumularse |
| **Escudo no funcional** | `shield_active` se activa pero `game_engine.py` no lo consulta para bloquear goles |
| **Sin movimiento del powerup** | Las esferas son estáticas; sin movimiento aleatorio por el campo |
| **Sin notificaciones** | No hay efectos de partículas ni sonidos específicos por tipo de powerup |
| **Duración no visible** | No se muestra ningún contador de tiempo restante junto al cronómetro de la partida |

---

## 2. Arquitectura del Nuevo Sistema

### 2.1 Visión General

El nuevo sistema sigue el patrón **Componente-Efecto** inspirado en ECS (Entity Component System). Cada powerup es una **definición de datos pura** registrada en un `PowerUpRegistry` centralizado en `shared/`. El `PowerUpManager` es un motor genérico que aplica, actualiza y elimina efectos sin necesidad de conocer la lógica interna de cada powerup.

```
shared/
└── powerups/
    ├── __init__.py
    ├── registry.py          ← PowerUpRegistry + PowerUpDefinition
    ├── effect_stack.py      ← EffectStack (pila por jugador)
    ├── components.py        ← Componentes de efectos reutilizables
    ├── definitions/
    │   ├── __init__.py
    │   ├── phase1_speed.py
    │   ├── phase2_shield.py
    │   ├── phase3_magnet.py
    │   ├── phase4_duplicate.py
    │   ├── phase5_slow.py
    │   ├── phase6_obstacle.py
    │   ├── phase7_paralyze.py
    │   └── phase8_invisibility.py
    └── manager.py           ← PowerUpManager (genérico)

game/
└── entities/
    └── powerup_renderer.py  ← PowerUpSphere (visual, partículas)
```

### 2.2 Contrato de una `PowerUpDefinition`

```python
# shared/powerups/registry.py
from dataclasses import dataclass, field
from typing import Callable, Optional

@dataclass
class PowerUpDefinition:
    id: str                          # Identificador único, ej: "speed_boost"
    name: str                        # Nombre temático, ej: "Viento Solar"
    description: str                 # Descripción breve para HUD
    help_text: str                   # Texto completo para botón de ayuda
    strategy_tip: str                # Tip de uso para el botón de ayuda
    duration: float                  # Duración configurable en segundos
    color: tuple                     # RGB para la esfera
    icon_char: str                   # Emoji o carácter para el ícono
    phase: int                       # Fase (1–8) en la que se desbloquea
    
    # Componentes de efecto (callables inyectados)
    on_collect: Callable             # (collector_idx, players, puck, state) → None
    on_expire: Callable              # (target_idx, players, puck, state) → None
    on_tick: Optional[Callable] = None  # (dt, target_idx, players, puck, state) → None (efectos por frame)
    
    # Metadatos
    affects_self: bool = True        # False = afecta al oponente
    can_stack: bool = True           # Puede acumularse con otros powerups del mismo tipo
    sound_key: str = ""              # Clave de sonido para AudioManager
    particle_color: tuple = field(default_factory=lambda: (255, 255, 255))
```

### 2.3 Sistema de Pila de Efectos (`EffectStack`)

Cada jugador tiene su propio `EffectStack`. Los efectos se acumulan, no se sobreescriben:

```python
# shared/powerups/effect_stack.py
class ActiveEffect:
    definition: PowerUpDefinition
    remaining: float
    stacked_multiplier: float = 1.0   # Modificado por powerup Duplicación

class EffectStack:
    """Pila de efectos activos para un jugador."""
    effects: list[ActiveEffect]
    
    def get_speed_multiplier(self) -> float:
        """Producto acumulado de todos los modificadores de velocidad."""
    
    def get_size_multiplier(self) -> float:
        """Producto acumulado de todos los modificadores de tamaño."""
    
    def has_shield(self) -> bool: ...
    def has_magnet(self) -> bool: ...
    def is_paralyzed(self) -> bool: ...
    def is_invisible(self) -> bool: ...
```

### 2.4 `PowerUpManager` Genérico

```python
# shared/powerups/manager.py
class PowerUpManager:
    def __init__(self, config: GameConfig, registry: PowerUpRegistry):
        self.stacks = [EffectStack(), EffectStack()]   # [jugador0, jugador1]
        self.field_powerups: list[PowerUpSphere] = []
        ...

    def update(self, dt, players, puck, state):
        self._spawn_logic(dt)
        self._update_field_powerups(dt)          # movimiento aleatorio + expiración
        self._check_collection(players, puck)    # colisión → activate
        self._tick_active_effects(dt, players, puck, state)
        self._expire_effects(players, puck, state)
        self._apply_magnet(players, puck)
        self._apply_paralysis(players)
        self._apply_obstacle()
```

### 2.5 Capa de Compatibilidad de Entrenamiento

```python
# training/envs/powerups_adapter.py
from shared.powerups.manager import PowerUpManager
from shared.powerups.registry import PowerUpRegistry

class TrainingPowerUpAdapter:
    """Adapta PowerUpManager al espacio de observación del agente RL."""
    def get_observation_vector(self) -> np.ndarray:
        """Genera el vector de obs compatible con el agente: N_powerups + N_efectos_activos."""
    
    def compute_reward_delta(self) -> float:
        """Retorna el delta de recompensa por colectar/perder powerups."""
```

---

## 3. Catálogo de Powerups por Fase

### Convención de colores y duración

| Rango de Fases | Duración base | Tipo |
|---|---|---|
| Fase 1–3 | **5 segundos** | Efectos de apoyo básicos |
| Fase 4–6 | **7 segundos** | Efectos estratégicos medios |
| Fase 7–8 | **3 segundos** | Efectos disruptivos fuertes |

> Todas las duraciones son configurables por nivel en `shared/powerups/definitions/*.py`.

---

### Fase 1 — Powerup de Velocidad

| Campo | Valor |
|---|---|
| **ID** | `speed_boost` |
| **Nombre temático** | "Viento Solar" |
| **Descripción HUD** | "+30% fuerza de golpe" |
| **Color esfera** | Cyan vibrante `(0, 200, 255)` |
| **Ícono** | ⚡ |
| **Duración** | 5 segundos |
| **Objetivo** | Propio jugador |
| **Efecto** | Aumenta `speed_multiplier` en `+30%` (multiplicador `1.3`) respecto a la base actual |
| **Composición** | Se apila con otras instancias multiplicando: `1.3 × 1.3 = 1.69` |
| **Expiración** | Elimina la contribución de este efecto del stack; otros efectos de velocidad persisten |
| **Efecto secundario** | Ninguno |
| **Estrategia recomendada** | "Recoger justo antes de un ataque para maximizar la fuerza del golpe. Combinado con el Imán, da control total del puck." |
| **Ayuda HUD** | "El viento solar carga tu mazo de energía cinética, incrementando la potencia de cada golpe. Cronometra su uso para los momentos de ataque." |

---

### Fase 2 — Powerup de Escudo

| Campo | Valor |
|---|---|
| **ID** | `shield` |
| **Nombre temático** | "Barrera de Ozono" |
| **Descripción HUD** | "Portería protegida" |
| **Color esfera** | Naranja-coral `(255, 130, 60)` |
| **Ícono** | 🛡 |
| **Duración** | 5 segundos |
| **Objetivo** | Propio jugador |
| **Efecto** | Despliega una barrera visible en la portería propia que intercepta el puck y lo rebota hacia el campo. La barrera se renderiza como un arco coral semitransparente con animación de pulso. |
| **Composición** | Dos escudos activos no se acumulan (el segundo reinicia el timer). |
| **Expiración** | La barrera desaparece; el puck recupera libre paso por la portería. |
| **Efecto secundario** | Si el puck ya cruzó la línea de gol antes de activarse el escudo, el gol cuenta. |
| **Estrategia recomendada** | "Activarlo cuando el oponente está en posición de ataque directo. No protege esquinas extremas." |
| **Ayuda HUD** | "La capa de ozono forma una barrera temporal que protege tu portería. El escudo rebota cualquier puck que intente atravesarla durante su duración." |

---

### Fase 3 — Powerup de Imán

| Campo | Valor |
|---|---|
| **ID** | `magnet` |
| **Nombre temático** | "Campo Magnético Terrestre" |
| **Descripción HUD** | "Atrae el puck (radio +8px)" |
| **Color esfera** | Púrpura `(170, 50, 220)` |
| **Ícono** | 🧲 |
| **Duración** | 5 segundos |
| **Objetivo** | Propio jugador |
| **Efecto** | Aplica una fuerza de atracción constante al puck en dirección al mazo del jugador. El área de influencia visual se extiende **8 píxeles de radio extra** con respecto al radio base del mazo, representada como un anillo pulsante animado de color púrpura translúcido. |
| **Composición** | Se apila con Velocidad: el puck llega más rápido al mazo. |
| **Expiración** | La fuerza de atracción se elimina; el anillo visual desaparece. |
| **Animación especial** | El anillo de radio extra pulsa (escala entre `0.9` y `1.1`) a 2 Hz. El grosor del anillo oscila con el tiempo restante (más grueso cuanto más tiempo queda). |
| **Efecto secundario** | Puede interferir con el movimiento del puck si el jugador está en el lado del oponente. |
| **Estrategia recomendada** | "Combinado con Velocidad Solar, permite controlar el puck y golpear con máxima potencia. En defensa, aleja el puck de tu portería pasivamente." |
| **Ayuda HUD** | "El campo magnético terrestre crea un aura de atracción alrededor de tu mazo. El anillo animado indica el área de influencia activa." |

---

### Fase 4 — Powerup de Duplicación

| Campo | Valor |
|---|---|
| **ID** | `duplication` |
| **Nombre temático** | "Efecto Invernadero" |
| **Descripción HUD** | "Duplica efectos activos" |
| **Color esfera** | Dorado `(255, 210, 0)` |
| **Ícono** | ✨ |
| **Duración** | 7 segundos |
| **Objetivo** | Propio jugador |
| **Efecto** | Aplica un `stacked_multiplier = 2.0` a todos los efectos activos del `EffectStack` del jugador en el momento de la colección. Los efectos futuros que se colecten durante su vigencia también se duplican en sus valores numéricos (multiplicadores de velocidad, fuerza, etc.). La **duración** de los otros powerups **no** se extiende. |
| **Composición** | Un segundo Efecto Invernadero aplica `× 4.0` (se acumula). |
| **Expiración** | El `stacked_multiplier` vuelve a `1.0` para todos los efectos; sus valores se reducen a sus valores base. |
| **Efecto secundario** | Un Imán duplicado puede ser excesivamente poderoso; considerar un cap máximo de `2.0` en fuerza de atracción. |
| **Estrategia recomendada** | "Recoger después de obtener Velocidad Solar o Imán para maximizar el combo. Es el powerup más valioso del campo." |
| **Ayuda HUD** | "El efecto invernadero amplifica todos tus poderes activos. Como los gases de efecto invernadero que amplifican el calor, este powerup duplica el impacto de cada beneficio que ya posees." |

---

### Fase 5 — Powerup de Lentitud

| Campo | Valor |
|---|---|
| **ID** | `slow_opponent` |
| **Nombre temático** | "Niebla Contaminante" |
| **Descripción HUD** | "-25% velocidad rival" |
| **Color esfera** | Gris humo `(130, 115, 100)` |
| **Ícono** | 🌫️ |
| **Duración** | 7 segundos |
| **Objetivo** | Oponente |
| **Efecto** | Reduce el `speed_multiplier` del oponente en `-25%` (multiplicador `0.75`). El mazo del oponente se torna visualmente semi-opaco (alpha `180`) durante el efecto. |
| **Composición** | Varias instancias se acumulan: `0.75 × 0.75 = 0.56` (cap mínimo: `0.4` para evitar inmovilización). |
| **Expiración** | El multiplicador vuelve a su valor base anterior al efecto. |
| **Efecto secundario** | En PvP, el oponente visualiza un aura gris sobre su mazo. En IA, el agente RL debe aprender a compensar la penalización de velocidad. |
| **Estrategia recomendada** | "Ideal para fases de ataque sostenido. Combinado con Velocidad Solar propia, la diferencia de velocidad es máxima." |
| **Ayuda HUD** | "La niebla contaminante envuelve al oponente, ralentizando su mazo. Como la polución que frena el progreso, este powerup reduce la movilidad rival durante 7 segundos." |

---

### Fase 6 — Powerup de Obstáculo

| Campo | Valor |
|---|---|
| **ID** | `obstacle` |
| **Nombre temático** | "Iceberg Flotante" |
| **Descripción HUD** | "Obstáculos temporales en campo" |
| **Color esfera** | Azul glacial `(180, 230, 255)` |
| **Ícono** | 🧊 |
| **Duración** | 7 segundos |
| **Objetivo** | Campo de juego (neutral) |
| **Efecto** | Despliega **2 obstáculos circulares** de radio `20px` en posiciones aleatorias del campo central (zona `30%–70%` horizontal, `20%–80%` vertical). El puck rebota con elasticidad completa (`1.0`) al colisionar con los obstáculos. Los mallets pueden pasar a través de ellos. Los obstáculos se renderizan como bloques de hielo translúcidos con efecto de brillo pulsante. |
| **Composición** | Una segunda instancia añade 2 obstáculos adicionales (máximo 6 en campo simultáneamente). |
| **Expiración** | Los obstáculos se desvanecen con animación de fundido (`alpha 0` en 0.5s). |
| **Efecto secundario** | Puede ocurrir que el puck quede atrapado entre dos obstáculos. Se incluye un mecanismo de "empuje suave" si el puck está inmóvil por más de 1 segundo. |
| **Estrategia recomendada** | "Activar cuando el oponente tiene control del campo. Los obstáculos desvían el puck de trayectorias predecibles." |
| **Ayuda HUD** | "El iceberg flotante coloca bloques de hielo en el campo que desvían el puck. Como los icebergs que obstruyen las rutas marítimas, estos obstáculos fuerzan nuevas estrategias de juego." |

---

### Fase 7 — Powerup de Parálisis

| Campo | Valor |
|---|---|
| **ID** | `paralyze` |
| **Nombre temático** | "Tormenta Eléctrica" |
| **Descripción HUD** | "Oponente paralizado 3s" |
| **Color esfera** | Amarillo eléctrico `(255, 240, 0)` |
| **Ícono** | ⚡⚡ |
| **Duración** | 3 segundos |
| **Objetivo** | Oponente |
| **Efecto** | Paraliza completamente el mazo del oponente (velocidad `0`, inputs ignorados). Se superpone una animación de rayo eléctrico sobre el mazo del oponente (arcos de electricidad generados proceduralmente cada 0.1s). El jugador paralizado ve el efecto sobre su propio mazo. |
| **Composición** | Una segunda instancia extiende la parálisis sumando duraciones. |
| **Expiración** | El mazo del oponente recupera el movimiento inmediatamente. |
| **Efecto secundario en IA:** | Si el agente RL está paralizado, se le entrega acción nula durante la parálisis. La observación incluye `is_paralyzed = 1` para que el agente aprenda el contexto. |
| **Estrategia recomendada** | "Activar y atacar directamente. Tienes 3 segundos de ventaja táctica absoluta." |
| **Ayuda HUD** | "La tormenta eléctrica descarga sobre el mazo del oponente, dejándolo sin control. Aprovecha estos 3 segundos para anotar goles sin resistencia." |

---

### Fase 8 — Powerup de Invisibilidad

| Campo | Valor |
|---|---|
| **ID** | `invisibility` |
| **Nombre temático** | "Capa de Invisibilidad" |
| **Descripción HUD** | "Tu mazo es invisible al rival" |
| **Color esfera** | Blanco fantasma `(220, 220, 255)` |
| **Ícono** | 👁️ |
| **Duración** | 3 segundos |
| **Objetivo** | Propio jugador |
| **Efecto** | El mazo del jugador activo se renderiza en la pantalla del **oponente** con `alpha = 20` (casi invisible). En PvP, se ocultará usando el canal alpha del sprite. En jugador vs IA, el agente RL no incluirá las coordenadas del mazo humano en su vector de observación (se reemplazan por ruido gaussiano). El jugador invisible se ve a sí mismo con `alpha = 150` (semi-transparente) para recordar que está activo. |
| **Composición** | Dos instancias activan "super-invisibilidad": `alpha = 10` sin ruido gaussiano. |
| **Expiración** | El mazo recupera `alpha = 255` con transición de `0.3s`. |
| **Efecto secundario** | El puck sigue siendo visible para ambos jugadores; solo el mazo del portador resulta afectado. |
| **Estrategia recomendada** | "Efectivo para movimientos de ataque fintas. El oponente no puede predecir tu posición exacta." |
| **Ayuda HUD** | "Como la capa de ozono que oculta la Tierra de la radiación, este powerup te hace casi imperceptible para el oponente durante 3 segundos. Úsalo para ataques sorpresa." |

---

## 4. Sistema de Representación Visual

### 4.1 Esferas de Powerup (`PowerUpSphere`)

Cada powerup en el campo se representa como una **esfera con movimiento aleatorio** que desaparece a los **5 segundos** si nadie la recoge.

```python
# game/entities/powerup_renderer.py
class PowerUpSphere(pygame.sprite.Sprite):
    FIELD_LIFETIME = 5.0  # segundos antes de desaparecer sola
    
    def __init__(self, definition: PowerUpDefinition, x, y, config):
        self.velocity = [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)]  # px/frame
        self.age = 0.0
        self.alpha = 255
        ...
    
    def update(self, dt, field_bounds):
        # Movimiento aleatorio: rebota suavemente en paredes internas
        self._move(dt)
        # Parpadeo cuando queda menos de 2s (warning visual)
        if self.age > self.FIELD_LIFETIME - 2.0:
            self.alpha = int(255 * abs(math.sin(self.age * 8)))
        # Expiración con fundido
        if self.is_expired():
            self._fade_out()
```

**Velocidad de movimiento:** `±1.5 px/frame` en X e Y, con rebote suave en las paredes del campo (no las porterías).  
**Comportamiento:** La esfera cambia de dirección aleatoriamente cada `2 ± 0.5` segundos para evitar trayectorias predecibles.

### 4.2 Efectos de Activación

Al recoger un powerup, se emiten **15 partículas** en ráfaga radial desde el punto de colisión:

| Parámetro | Valor |
|---|---|
| Conteo | 15 partículas |
| Color | Color del powerup recogido |
| Velocidad | `random.uniform(2, 5)` px/frame |
| Ángulo | Aleatorio en 360° |
| Vida | `0.4` segundos |
| Forma | Círculo de radio 3 |

### 4.3 Efectos Visuales Continuos por Tipo

| Powerup | Efecto visual en mazo |
|---|---|
| Velocidad | Estela de cyan tras el movimiento (trail de 5 frames) |
| Escudo | Arco naranja semitransparente en la portería propia |
| Imán | Anillo pulsante púrpura alrededor del mazo (radio base + 8px) |
| Duplicación | Destello dorado intermitente cada 0.5s |
| Lentitud | Mazo del oponente con halo gris humo |
| Obstáculo | Bloques de hielo azulados en el campo |
| Parálisis | Arcos de electricidad amarillos sobre el mazo del oponente |
| Invisibilidad | Mazo con alpha reducido (visible para el portador, casi invisible para el rival) |

---

## 5. Sistema de Notificaciones

### 5.1 Notificaciones Visuales

```python
# game/components/PowerUpNotification.py
class PowerUpNotification:
    """Toast animado que aparece cuando se activa un powerup."""
    DISPLAY_DURATION = 2.5  # segundos
    
    def __init__(self, definition: PowerUpDefinition, player_side: str):
        # Texto: "{icono} {nombre}\n{descripcion}"
        # Fondo: panel semitransparente del color del powerup
        # Animación: slide-in desde el borde superior del campo
        ...
```

**Posición:** Panel flotante en la zona superior del campo, diferenciado por lado (izquierda para jugador 1, derecha para jugador 2).  
**Animación:** Desliza desde arriba en `0.3s`, permanece `2s`, desvanece en `0.2s`.

### 5.2 Notificaciones Sonoras

| Powerup | Sonido sugerido | Archivo |
|---|---|---|
| Velocidad | Ráfaga de viento | `sounds/powerup_speed.wav` |
| Escudo | Activación de barrera | `sounds/powerup_shield.wav` |
| Imán | Pulso magnético | `sounds/powerup_magnet.wav` |
| Duplicación | Incremento tonal | `sounds/powerup_duplicate.wav` |
| Lentitud | Niebla / opresivo | `sounds/powerup_slow.wav` |
| Obstáculo | Crujido de hielo | `sounds/powerup_obstacle.wav` |
| Parálisis | Descarga eléctrica | `sounds/powerup_paralyze.wav` |
| Invisibilidad | Whoosh suave | `sounds/powerup_invisible.wav` |
| Expiración (todos) | Tono descendente | `sounds/powerup_expire.wav` |

> Los sonidos se registran en `AudioManager` bajo claves `powerup_{id}` y `powerup_expire`.

---

## 6. Integración con HUD

### 6.1 Temporizadores de Powerup junto al Cronómetro

Los powerups activos se muestran como pastillas de color **al lado del cronómetro de la partida** en la barra central del HUD.

```
[  JUGADOR  2 - 1  OPONENTE  |  ⚡2.3s  🧲4.1s  |  03:45  ]
```

**Formato por pastilla:**
- Fondo de color del powerup con alpha `200`
- Ícono del powerup (icono de l powerup gestionado por pygame)
- Tiempo restante en segundos con 1 decimal
- Cuando queda < 1s: borde pulsante rojo y texto en rojo

### 6.2 Indicadores de Efectos del Oponente

Los efectos que afectan al oponente (Lentitud, Parálisis) se muestran en el **lado del oponente** del HUD como íconos "negativos" con borde rojo para diferenciarse.

### 6.3 Modificaciones en `game/ui/hud.py`

- Nuevo método `_draw_powerup_timers(screen, stacks, W, H)` que reemplaza `_draw_powerup_indicators`.
- Integración con el layout del bloque central `_draw_score_and_timer`.
- Soporte para mostrar hasta **4 powerups activos simultáneamente** por jugador antes de colapsar en contador `+N`.

---

## 7. Integración con Entornos de Entrenamiento

### 7.1 Adaptador Unificado

Se elimina `training/envs/powerups_env.py` en favor del adaptador compartido:

```python
# training/envs/powerups_env.py  ← REFACTORIZADO
from shared.powerups.manager import PowerUpManager
from training.envs.powerups_adapter import TrainingPowerUpAdapter

class AirHockeyWithPowerUpsEnv(AirHockeyEnv):
    def __init__(self, ...):
        self.powerup_manager = PowerUpManager(config, registry)
        self.powerup_adapter = TrainingPowerUpAdapter(self.powerup_manager)
        ...
```

### 7.2 Espacio de Observación Expandido

El vector de observación del agente RL se amplía de forma modular. Cada fase de powerups puede activarse/desactivarse mediante flags:

```python
# Observación base: 13 dims (igual que ahora)
# + Fase 1-3 activas: +8 dims (posición y tipo de 2 powerups en campo + 4 efectos del agente)
# + Fase 4-6 activas: +4 dims adicionales
# + Fase 7-8 activas: +2 dims (is_paralyzed, is_invisible_to_me)
# Total máximo: 27 dims
```

La dimensión final se determina en `TrainingPowerUpAdapter.get_obs_size()` para facilitar el ajuste por fase sin romper modelos entrenados en fases anteriores.

### 7.3 Recompensas por Powerup para el Agente RL

```python
# training/envs/rewards.py  ← añadir sección PowerupRewards
REWARD_COLLECT_POSITIVE_POWERUP   = +0.15   # recoger powerup que afecta al propio jugador
REWARD_COLLECT_NEGATIVE_POWERUP   = +0.10   # recoger powerup que afecta al oponente
REWARD_EVADE_OBSTACLE              = +0.05   # puck esquiva obstáculo generado
REWARD_OPPONENT_PARALYZED_GOAL    = +0.25   # gol mientras el oponente está paralizado
```

---

## 8. Integración con el Botón de Ayuda

### 8.1 Panel de Ayuda de Powerups

El botón de ayuda existente en el juego (o un nuevo sub-panel "Powerups") mostrará una tarjeta por cada powerup:

```
╔══════════════════════════════════╗
║  ⚡  Viento Solar                 ║
║  Duración: 5s                     ║
║  "+30% fuerza de golpe"          ║
║                                   ║
║  El viento solar carga tu mazo   ║
║  de energía cinética...          ║
║                                   ║
║  💡 Úsalo antes de atacar para   ║
║  maximizar la potencia del golpe. ║
╚══════════════════════════════════╝
```

### 8.2 Estructura del Componente de Ayuda

```python
# game/components/PowerUpHelpPanel.py
class PowerUpHelpPanel:
    def __init__(self, registry: PowerUpRegistry, config: GameConfig):
        self.cards = [PowerUpHelpCard(d) for d in registry.get_all_by_phase()]
    
    def draw(self, screen, scroll_offset):
        """Dibuja las tarjetas con scroll si hay muchas."""
```

Los campos `help_text` y `strategy_tip` de cada `PowerUpDefinition` alimentan directamente este panel.

---

## 9. Plan de Fases e Implementación por Tarea

### Fase 0 — Infraestructura Base (Prerequisito)

| ID | Tarea | Descripción | Prioridad |
|---|---|---|---|
| T-0.1 | Crear estructura `shared/powerups/` | Directorios, `__init__.py`, `registry.py` con `PowerUpDefinition` y `PowerUpRegistry` | 🔴 Crítica |
| T-0.2 | Implementar `effect_stack.py` | `ActiveEffect` y `EffectStack` con getters de multiplicadores acumulados | 🔴 Crítica |
| T-0.3 | Implementar `shared/powerups/manager.py` genérico | Sin lógica de powerup específico; loop de spawn, update, tick, expire | 🔴 Crítica |
| T-0.4 | Crear `game/entities/powerup_renderer.py` | `PowerUpSphere` con movimiento aleatorio, animación de pulso y expiración a 5s | 🔴 Crítica |
| T-0.5 | Sistema de partículas de activación | `ParticleSystem` simple con emisión radial en colección | 🟡 Alta |
| T-0.6 | Refactorizar `game/ui/hud.py` | Nuevo método `_draw_powerup_timers` usando `EffectStack`; eliminar método viejo | 🔴 Crítica |
| T-0.7 | Refactorizar `game/core/game_engine.py` | Usar nuevo `PowerUpManager` de `shared/`; eliminar referencias al viejo | 🔴 Crítica |
| T-0.8 | `game/components/PowerUpNotification.py` | Toast animado de activación/expiración | 🟡 Alta |
| T-0.9 | `game/components/PowerUpHelpPanel.py` | Panel de ayuda scrollable con tarjetas por powerup | 🟡 Alta |
| T-0.10 | Registrar sonidos en `AudioManager` | Añadir claves `powerup_{id}` y `powerup_expire` | 🟢 Media |
| **T-0.11** | **Eliminar código viejo** | Borrar `game/entities/powerups.py` y `training/envs/powerups_env.py` (versión antigua) | 🔴 Crítica |

### Fase 1 — Powerup de Velocidad

| ID | Tarea | Descripción |
|---|---|---|
| T-1.1 | Implementar `shared/powerups/definitions/phase1_speed.py` | `on_collect`, `on_expire`, composición en `EffectStack.get_speed_multiplier()` |
| T-1.2 | Efectos visuales de velocidad en mazo | Trail de estela cyan en `powerup_renderer.py` |
| T-1.3 | Tests unitarios Fase 1 | Verificar multiplicador, composición con 2 instancias, expiración correcta |

### Fase 2 — Powerup de Escudo

| ID | Tarea | Descripción |
|---|---|---|
| T-2.1 | Implementar `shared/powerups/definitions/phase2_shield.py` | `on_collect`, `on_expire`; `on_tick` renderiza el arco en portería |
| T-2.2 | Integrar detección de escudo en `game_engine.py` | En `_check_goal()`, consultar `stack.has_shield()` antes de registrar gol |
| T-2.3 | Visual del arco de escudo | Arco semitransparente naranja con pulso animado en portería |
| T-2.4 | Tests Fase 2 | Verificar bloqueo de gol, no bloqueo de gol cuando está inactivo |

### Fase 3 — Powerup de Imán

| ID | Tarea | Descripción |
|---|---|---|
| T-3.1 | Implementar `shared/powerups/definitions/phase3_magnet.py` | `on_tick` aplica fuerza de atracción de `0.3 px/frame`, con cap en `dist > 5` |
| T-3.2 | Anillo pulsante animado | Renderizar anillo de `radius + 8px` pulsante en `powerup_renderer.py` |
| T-3.3 | Tests Fase 3 | Verificar atracción funcional, que el anillo aparece/desaparece correctamente |

### Fase 4 — Powerup de Duplicación

| ID | Tarea | Descripción |
|---|---|---|
| T-4.1 | Implementar `shared/powerups/definitions/phase4_duplicate.py` | `on_collect` aplica `stacked_multiplier = 2.0` a todos los `ActiveEffect` del `EffectStack` |
| T-4.2 | Modificar `EffectStack` para soporte de `stacked_multiplier` | Getters deben considerar: `base_mult × stacked_multiplier` |
| T-4.3 | Visual de duplicación | Destello dorado intermitente sobre el mazo |
| T-4.4 | Tests Fase 4 | Verificar composición correcta con Velocidad, Imán y Lentitud |

### Fase 5 — Powerup de Lentitud

| ID | Tarea | Descripción |
|---|---|---|
| T-5.1 | Implementar `shared/powerups/definitions/phase5_slow.py` | Afecta al oponente; cap mínimo de `0.4` en `get_speed_multiplier()` |
| T-5.2 | Visual en mazo del oponente | Halo gris humo (alpha overlay) |
| T-5.3 | Tests Fase 5 | Cap mínimo funcional, composición de múltiples instancias |

### Fase 6 — Powerup de Obstáculo

| ID | Tarea | Descripción |
|---|---|---|
| T-6.1 | Implementar `shared/powerups/definitions/phase6_obstacle.py` | `on_collect` crea 2 `Obstacle` en el campo; `on_expire` los elimina con fundido |
| T-6.2 | Clase `Obstacle` | Colisión con el puck (rebote elástico), sin colisión con mallets |
| T-6.3 | Integrar en `game_engine.py` | Verificar colisiones puck-obstáculo en `_update_playing()` |
| T-6.4 | Visual de obstáculos | Bloques hexagonales azul glacial con brillo pulsante |
| T-6.5 | Mecanismo anti-atasco | Si puck inmóvil > 1s, aplicar empuje suave aleatorio |
| T-6.6 | Tests Fase 6 | Colisión puck-obstáculo, generación/eliminación, anti-atasco |

### Fase 7 — Powerup de Parálisis

| ID | Tarea | Descripción |
|---|---|---|
| T-7.1 | Implementar `shared/powerups/definitions/phase7_paralyze.py` | `on_collect` pone `is_paralyzed = True` en oponente; `on_tick` bloquea inputs |
| T-7.2 | Modificar `HumanMallet.update()` | Si el stack del jugador tiene `is_paralyzed`, ignorar mouse_pos |
| T-7.3 | Modificar `KeyboardMallet.update()` | Misma protección |
| T-7.4 | Modificar lógica RL en `game_engine.py` | Si agente paralizado, pasar acción de "quieto" (acción 4) |
| T-7.5 | Animación de electricidad | Arcos procedurales amarillos sobre el mazo del oponente cada 0.1s |
| T-7.6 | Tests Fase 7 | Oponente inmóvil durante duración, recuperación correcta |

### Fase 8 — Powerup de Invisibilidad

| ID | Tarea | Descripción |
|---|---|---|
| T-8.1 | Implementar `shared/powerups/definitions/phase8_invisibility.py` | `is_invisible` en stack; `on_tick` ajusta alpha del sprite en renderer |
| T-8.2 | Modificar `Renderer` para alpha condicional | Si el jugador contrario tiene escudo, renderizar mazo con alpha bajo |
| T-8.3 | Modificar observación RL | Si humano invisible, reemplazar coordenadas por ruido gaussiano `N(pos, 0.05)` |
| T-8.4 | Visual de autoconciencia | Mazo propio en alpha `150` para recordar al portador que está activo |
| T-8.5 | Tests Fase 8 | Verificar alpha en renderer, perturbación de obs RL |

---

## 10. Asignación de Responsabilidades por Subagente/Skill

| Área | Subagente / Skill | Tareas |
|---|---|---|
| **Arquitectura central** | `python-design-patterns` | T-0.1, T-0.2, T-0.3 — Diseño de `PowerUpRegistry`, `EffectStack`, `PowerUpManager` genérico siguiendo KISS y SRP |
| **Sistemas de juego (pygame)** | `pygame-patterns` | T-0.4, T-0.5, T-3.2, T-4.3, T-7.5 — `PowerUpSphere`, partículas, efectos visuales de mazo, anillo del imán |
| **Game Feel y Feedback** | `game-designer` | T-0.8, T-5.2, T-7.5, T-8.4 — Toasts de notificación, partículas de activación, efectos visuales de duración para sensación de impacto |
| **UI del juego** | `game-ui-design` | T-0.6, T-0.9, T-6.4 — HUD de temporizadores, panel de ayuda scrollable, diseño visual de pastillas de efecto y tarjetas de ayuda |
| **Integración de motor** | `game-developer` | T-0.7, T-2.2, T-6.3, T-7.4, T-8.3 — Integración en `game_engine.py`, `HumanMallet`, `KeyboardMallet`, `Renderer` |
| **Entrenamiento RL** | `stable-baselines3` | T-7.4, T-8.3, Fase 7-8 obs space — Modificación del espacio de observación, recompensas, adaptador de entrenamiento |
| **Definiciones de powerups** | `game-developer` | T-1.1 a T-8.1 — Implementar los 8 archivos `phaseN_*.py` con `on_collect`, `on_expire`, `on_tick` |
| **Sonido** | `game-developer` | T-0.10 — Registro de sonidos en `AudioManager`; asset audio requerido del equipo de diseño |
| **Tests** | `python-design-patterns` | T-1.3, T-2.4, T-3.3, T-4.4, T-5.3, T-6.6, T-7.6, T-8.5 — Tests unitarios e integración por fase |

---

## 11. Criterios de Aceptación y Tests

### 11.1 Tests Unitarios (`tests/test_powerups.py`)

```python
# Estructura esperada
class TestEffectStack:
    def test_speed_accumulation()       # 2 instancias → multiplier = 1.3 × 1.3
    def test_speed_cap_slow()           # lentitud no baja de 0.4
    def test_duplication_doubles()      # Duplicación × 2 sobre velocidad
    def test_shield_blocks_goal()       # Gol rechazado con escudo activo
    def test_paralyze_blocks_input()    # HumanMallet.update() ignorado
    def test_magnet_attraction()        # Puck se acerca en < 3s
    def test_obstacle_puck_bounce()     # Puck rebota en obstáculo
    def test_invisibility_alpha()       # Sprite alpha < 50 en renderer rival
    def test_expire_removes_effect()    # Stack vacío tras expiración

class TestPowerUpSphere:
    def test_sphere_expires_in_5s()    # Age >= 5.0 is_expired = True
    def test_sphere_moves_randomly()   # posición diferente tras update
    def test_collection_collides()     # check_collision True dentro del radio
```

### 11.2 Tests de Integración (`tests/test_integration.py`)

- Simular 300 steps de juego con powerups habilitados — sin crashes.
- Verificar que gol con escudo activo no incrementa `state.ai_score`.
- Verificar que puck con imán activo reduce distancia al mazo en 10 frames.
- Verificar que el vector de observación RL no excede la dimensión declarada.

### 11.3 Prueba Visual (QA)

- [ ] Esferas visibles y en movimiento en el campo.
- [ ] Esferas desaparecen a los 5 segundos si no se recogen.
- [ ] Toasts de activación aparecen correctamente.
- [ ] Temporizadores en HUD muestran tiempo restante con precisión.
- [ ] Anillo de imán pulsante visible.
- [ ] Escudo visible en portería.
- [ ] Obstáculos azules visibles y funcionales al rebotar el puck.
- [ ] Electricidad sobre mazo paralizado.
- [ ] Mazo invisible al rival durante invisibilidad.

---

## 12. Configuración y Variables de Ajuste

### 12.1 Archivo de Configuración Central

```python
# shared/powerups/config.py
POWERUP_CONFIGS = {
    "speed_boost":   {"duration": 5.0,  "multiplier": 1.3,  "phase": 1},
    "shield":        {"duration": 5.0,  "phase": 2},
    "magnet":        {"duration": 5.0,  "extra_radius": 8,  "force": 0.3, "phase": 3},
    "duplication":   {"duration": 7.0,  "stack_mult": 2.0,  "phase": 4},
    "slow_opponent": {"duration": 7.0,  "multiplier": 0.75, "min_cap": 0.4, "phase": 5},
    "obstacle":      {"duration": 7.0,  "count": 2, "radius": 20, "phase": 6},
    "paralyze":      {"duration": 3.0,  "phase": 7},
    "invisibility":  {"duration": 3.0,  "alpha": 20, "phase": 8},
}

SPHERE_FIELD_LIFETIME    = 5.0     # segundos de vida en campo sin ser recogida
SPHERE_SPEED             = 1.5     # px/frame velocidad de movimiento
SPHERE_DIRECTION_CHANGE  = 2.0     # segundos entre cambios de dirección
MAX_POWERUPS_ON_FIELD    = 3       # máximo simultáneo en campo
SPAWN_INTERVAL_MIN       = 8.0     # segundos mínimos entre spawns
SPAWN_INTERVAL_MAX       = 15.0    # segundos máximos entre spawns
PARTICLE_COUNT           = 15      # partículas en activación
```

### 12.2 Configuración por Nivel

Cada nivel puede sobrescribir los valores del `POWERUP_CONFIGS` de forma selectiva a través del `level_config`:

```python
# Ejemplo en game/config/level_config.py
{
    "level_id": 4,
    "powerups_enabled": True,
    "powerups_phases": [1, 2, 3, 4],  # Solo fases 1-4 disponibles en nivel 4
    "powerup_overrides": {
        "speed_boost": {"duration": 7.0},  # Más duración en este nivel
    }
}
```

---

## Apéndice: Decisiones de Diseño

| Decisión | Razón |
|---|---|
| `EffectStack` por jugador (no global) | Aísla efectos; facilita consultas en O(1) |
| Callables en `PowerUpDefinition` en lugar de herencia | Sigue composición > herencia (KISS); añadir powerup = añadir archivo, no modificar clase base |
| `shared/powerups/` en vez de `game/entities/` | Permite reutilización en entornos de entrenamiento sin importar `pygame` |
| Movimiento aleatorio de esferas a `1.5 px/frame` | Suficiente para visualización dinámica sin afectar la jugabilidad |
| Vida de esfera: 5 segundos | Crea urgencia de recolección sin hacer los powerups demasiado efímeros |
| Cap mínimo de velocidad: `0.4` con Lentitud | Evita inmovilización completa que resultaría injugable |
| Paralelismo de fases | Cada fase es independiente: se puede entregar y testear en PR separados |
