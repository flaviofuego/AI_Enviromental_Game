# Plan de Implementación — Mejoras Air Hockey Ambiental

> **Fecha:** 16 de febrero de 2026  
> **Versión:** 1.0  
> **Estado:** Pendiente de ejecución

---

## Índice

1. [Mejora 1: Reubicar botón Play y P1vsP2 en Home](#mejora-1)
2. [Mejora 2: Ventana de configuración PvP completa](#mejora-2)
3. [Mejora 3: Porterías correctas por nivel](#mejora-3)
4. [Mejora 4: Cronómetro visible durante partida](#mejora-4)
5. [Mejora 5: Rendimiento de UI — skins, perfiles, etc.](#mejora-5)
6. [Mejora 6: Componente Button universal optimizado](#mejora-6)
7. [Mejora 7: Rúbrica de rewards (ofensivo + defensivo)](#mejora-7)
8. [Mejora 8: Oponente de entrenamiento mejorado](#mejora-8)
9. [Mejora 9: Optimización de arquitectura del agente](#mejora-9)
10. [Mejora 10: Componentizar UI para evitar re-renderizados](#mejora-10)

---

## Mejora 1: Reubicar botón Play y P1vsP2 en Home {#mejora-1}

### Problema actual
En `game/pages/home.py`, el botón **Play** no lleva a la selección de niveles correctamente y el botón del modo **P1vsP2** debe reubicarse para no interferir con la navegación principal.

### Diagnóstico técnico
- **Archivo:** `game/pages/home.py` → clase `HockeyMainScreen`
- El método `handle_click()` (línea ~380) despacha `play` → `"level_select"` pero **solo si hay un perfil activo**; si no hay perfil, redirige internamente a la pantalla de perfiles sin feedback claro al usuario.
- El botón `pvp` → `"pvp_setup"` funciona, pero su posición visual puede confundir con el flujo principal.
- En `game/hub/screen_controller.py` → `_handle_home()`, el resultado del `run()` de `HockeyMainScreen` mapea `"level_select"` y `"pvp_setup"` correctamente.

### Acciones a implementar

| # | Tarea | Archivo(s) | Detalle |
|---|-------|-----------|---------|
| 1.1 | Auditar flujo actual del botón Play | `game/pages/home.py` L350-420 | Verificar que `handle_click` retorna `"level_select"` sin redirección oculta |
| 1.2 | Reorganizar layout de botones | `game/pages/home.py` → método `run()` donde se crean los `Button` | Mover botón PvP a una posición secundaria (ej: esquina inferior o fila separada), hacer que Play sea el botón dominante central |
| 1.3 | Añadir feedback visual cuando no hay perfil | `game/pages/home.py` → `handle_click()` | Mostrar tooltip/popup indicando "Crea un perfil antes de jugar" en vez de redirigir silenciosamente |
| 1.4 | Actualizar posiciones/coordenadas | `game/pages/home.py` | Recalcular las posiciones `(x, y)` de cada `Button` para el layout corregido |

### Skills / Herramientas
- **Skill:** `game-ui-design` → consultar `references/patterns.md` para layout de menú principal
- **Skill:** `pygame-patterns` → patrón de input discreto (evitar `get_pressed()` continuo)
- **MCP:** `mcp_context7_get-library-docs` → documentación de pygame para `pygame.event` vs `pygame.mouse.get_pressed()`

### Criterio de aceptación
- [ ] Botón Play lleva directamente a `Level_Select` cuando hay perfil activo
- [ ] Botón Play muestra popup si no hay perfil
- [ ] Botón PvP está visualmente separado del flujo principal
- [ ] Navegación back funciona correctamente desde ambas rutas

---

## Mejora 2: Ventana de configuración PvP completa {#mejora-2}

### Problema actual
La pantalla de configuración PvP (`game/pages/pvp_setup.py` → `PvPSetupScreen`) existe pero necesita agregar:
1. **Selección de background** de entre los 5 niveles
2. **Selección de skin** para ambos jugadores (ya parcialmente implementado con `SkinSelector`)
3. **Máximo de goles** (1, 2, 3, o más)
4. **Tiempo de duración** (5, 10, 15 minutos, o sin límite)

### Diagnóstico técnico
- **Archivo actual:** `game/pages/pvp_setup.py` → `PvPSetupScreen`
- Ya tiene: `SCORE_OPTIONS = [3, 5, 7, 10, 15]`, `TIME_OPTIONS = [0, 60, 120, 180, 300]` con flechas ◀▶
- Ya tiene `SkinSelector` para cada jugador con grid expandible
- **Falta:** selector de fondo/nivel (solo backgrounds) y la config de goles con valores (1, 2, 3, ...)
- El `get_config()` retorna `{score_limit, time_limit, powerups, p1_skin, p2_skin}` — **no incluye `level_id` para el background**
- En `screen_controller.py` → `_launch_pvp_game(config)`, se llama `start_game(self.screen, level_id=1, ...)` → **level_id=1 hardcodeado**

### Acciones a implementar

| # | Tarea | Archivo(s) | Detalle |
|---|-------|-----------|---------|
| 2.1 | Agregar selector de background | `game/pages/pvp_setup.py` | Crear sección "Escenario" con thumbnails de los 5 niveles usando `LevelThumbnail` existente. Click selecciona fondo. |
| 2.2 | Actualizar `SCORE_OPTIONS` | `game/pages/pvp_setup.py` | Cambiar a `[1, 2, 3, 5, 7, 10]` según requisito |
| 2.3 | Actualizar `TIME_OPTIONS` | `game/pages/pvp_setup.py` | Cambiar a `[0, 300, 600, 900]` (0=sin límite, 300=5min, 600=10min, 900=15min) |
| 2.4 | Agregar `background_level_id` a `get_config()` | `game/pages/pvp_setup.py` | Incluir en el dict retornado |
| 2.5 | Pasar `level_id` dinámico en PvP | `game/hub/screen_controller.py` → `_launch_pvp_game()` | Usar `config["background_level_id"]` en vez de `level_id=1` hardcodeado |
| 2.6 | Ajustar layout de la pantalla | `game/pages/pvp_setup.py` | Reestructurar layout: columna izquierda = escenario + opciones de partido, columna derecha = skins de jugadores |
| 2.7 | Previsualización del background seleccionado | `game/pages/pvp_setup.py` | Mostrar preview pequeño del background seleccionado |

### Skills / Herramientas
- **Skill:** `game-ui-design` → consultar `references/patterns.md` para diseño de pantallas de configuración
- **Skill:** `pygame-patterns` → patrones de sprite loading para thumbnails
- **Subagente:** investigar cómo implementar un carrusel horizontal de thumbnails en pygame (o usar grid 5x1)
- **Archivo de referencia:** `game/components/LevelThumbnail.py` → reutilizar para los thumbnails del selector de background
- **Archivo de referencia:** `game/components/Card.py` → adaptar para tarjetas de selección de background

### Criterio de aceptación
- [ ] Se puede seleccionar fondo de los 5 niveles existentes
- [ ] Se puede seleccionar skin de ambos jugadores
- [ ] Se puede configurar goles: 1, 2, 3, 5, 7, 10
- [ ] Se puede configurar tiempo: sin límite, 5 min, 10 min, 15 min
- [ ] La partida PvP usa el background del nivel seleccionado
- [ ] Las porterías del nivel seleccionado se cargan correctamente

---

## Mejora 3: Porterías correctas por nivel {#mejora-3}

### Problema actual
Todos los niveles usan la misma portería del nivel 1 en vez de sus propias porterías.

### Diagnóstico técnico
El código de carga de assets **es correcto en teoría**:

```
SpriteLoader.load_level_sprites(level_id, config)  # usa level_id correcto
→ get_asset_path(level_id, "porteria_izq.png")     # ruta correcta
→ game/assets/niveles/{level_id}/porteria_izq.png   # archivo existe para c/nivel
```

**Todos los 5 niveles tienen** sus propias porterías en `game/assets/niveles/{1-5}/porteria_izq.png` y `porteria_der.png`.

El bug podría estar en:
1. **`game_engine.py` L149-155:** `_setup_table()` carga porterías desde `self.assets` correctamente
2. **`table.py` L32:** `set_goal_sprites()` asigna los sprites
3. **Posible causa:** Los sprites de portería de niveles 2-5 podrían ser **copias del nivel 1** (archivos idénticos), o el `level_id` no se está pasando correctamente en ciertos flujos

### Acciones a implementar

| # | Tarea | Archivo(s) | Detalle |
|---|-------|-----------|---------|
| 3.1 | Verificar visualmente los sprites de portería | `game/assets/niveles/*/porteria_*.png` | Abrir y comparar los archivos PNG de cada nivel — ¿son diferentes o son el mismo archivo copiado? |
| 3.2 | Añadir logging de carga de porterías | `shared/utils/sprite_loader.py` L72-77 | Añadir `print(f"Loading goal sprite: {p}")` para verificar en runtime |
| 3.3 | Verificar `level_id` en flujo PvP | `game/hub/screen_controller.py` → `_launch_pvp_game()` | Confirmar que `level_id` se pasa correctamente (actualmente hardcodeado a 1) |
| 3.4 | Verificar `level_id` en flujo PvAI | `game/pages/Level_Select.py` → `start_level()` | Confirmar que el `level_id` seleccionado llega a `GameEngine` |
| 3.5 | Corregir la carga si los sprites son iguales | `game/assets/niveles/*/` | Crear porterías distintas si todos son el mismo archivo, o corregir la ruta si hay error de path |

### Skills / Herramientas
- **Terminal:** `md5sum game/assets/niveles/*/porteria_izq.png` para verificar si los archivos son idénticos
- **Skill:** `pygame-patterns` → patrón de sprite loading con error handling

### Criterio de aceptación
- [ ] Nivel 1 muestra porterías del nivel 1
- [ ] Nivel 2 muestra porterías del nivel 2
- [ ] Nivel 3-5 muestran sus porterías respectivas
- [ ] Porterías se ven correctamente escaladas en todos los niveles

---

## Mejora 4: Cronómetro visible durante la partida {#mejora-4}

### Problema actual
1. El cronómetro no aparece visiblemente durante la partida
2. El cronómetro aparece al finalizar la partida y **no se detiene** (sigue avanzando)

### Diagnóstico técnico
- **HUD:** `game/ui/hud.py` → `_draw_timer()` **sí dibuja el timer** como countdown (`MM:SS`), pero solo cuando `time_limit > 0`
- **Problema 1:** En modo PvAI, `match_config.time_limit_seconds` es `None` (no `0`), lo que hace que `_draw_timer()` no se ejecute
- **Problema 2:** El `elapsed` se calcula con `time.time() - match_start_time` en `game_engine.py`, pero `match_start_time` **no se resetea al pausar** ni se detiene en game over
- **Problema 3:** En la pantalla de game over (`game_engine.py` → `_draw_game_over()`), el timer sigue corriendo porque el loop principal sigue llamando `time.time() - match_start_time`

### Acciones a implementar

| # | Tarea | Archivo(s) | Detalle |
|---|-------|-----------|---------|
| 4.1 | Mostrar timer siempre (no solo countdown) | `game/ui/hud.py` → `_draw_timer()` | Si `time_limit` es `None` o `0`: mostrar tiempo transcurrido (`↑ MM:SS`). Si hay `time_limit > 0`: mostrar countdown (`↓ MM:SS`). Colocar fuera de la cancha (zona superior, junto al score) |
| 4.2 | Pausar el timer durante pausa | `game/core/game_state.py` | Añadir `pause_start_time` y `total_pause_duration`. Al pausar: guardar `time.time()`. Al reanudar: acumular diferencia. `get_elapsed()` retorna `time.time() - match_start_time - total_pause_duration` |
| 4.3 | Congelar timer en game over | `game/core/game_engine.py` | Al transicionar a `GAME_OVER`: guardar `self.state.final_elapsed = elapsed` y usar ese valor fijo en el draw del game over |
| 4.4 | Reposicionar timer visualmente | `game/ui/hud.py` → `_draw_timer()` | Colocarlo centrado debajo del marcador, **fuera de la cancha** de juego para no tapar la acción. Usar la zona del HUD superior |
| 4.5 | Mostrar formato legible | `game/ui/hud.py` | Timer grande con fuente clara, formato `MM:SS`, color neutro (blanco) que pase a rojo cuando queden < 30s |

### Skills / Herramientas
- **Skill:** `game-ui-design` → posicionamiento de timer en HUD (referencia: esports timer placement)
- **Skill:** `pygame-patterns` → delta_time y manejo de tiempo frame-independent
- **MCP:** `mcp_context7_get-library-docs` → `pygame.time` para manejo preciso de tiempo

### Criterio de aceptación
- [ ] Timer visible en todo momento durante la partida
- [ ] Timer en formato `MM:SS` ubicado fuera de la cancha
- [ ] Timer se pausa cuando el juego está en pausa
- [ ] Timer se congela en la pantalla de game over
- [ ] Timer muestra countdown cuando hay límite de tiempo
- [ ] Timer muestra tiempo transcurrido cuando no hay límite de tiempo
- [ ] Timer parpadea en rojo cuando quedan < 30 segundos

---

## Mejora 5: Rendimiento de UI — skins, perfiles y similares {#mejora-5}

### Problema actual
La UI se bloquea al hacer click en cualquier elemento de selección de skins, perfiles, y similares. El rendimiento es muy lento.

### Diagnóstico técnico (bugs identificados por severidad)

| Severidad | Archivo | Problema | Causa raíz |
|-----------|---------|----------|------------|
| **CRÍTICA** | `game/components/EnvironmentalEffects.py` | Crea `pygame.Surface(SRCALPHA)` por **cada partícula en cada frame** | ~89 surfaces creadas/destruidas por frame (20 base + 15 contaminación + 8 heat + 25 rain + 6 ice + 10 hope + 5 aurora) |
| **CRÍTICA** | `game/pages/home.py` L513-527 | `_handle_profile_events()` llama `draw_*()` del `ProfileManager` **solo para obtener los rects de UI**, causando **doble renderizado completo** | Mezcla de manejo de eventos con rendering |
| **ALTA** | `game/components/Card.py` L72 | **Crea fuente nueva** (`pygame.font.Font(None, size)`) en cada llamada a `draw()` | Sin cache de fuentes |
| **ALTA** | `game/components/AudioManager.py` | `time.sleep(0.5)` bloqueante en `play_music()` durante fade-out | Bloquea thread principal |
| **MEDIA** | `game/pages/Level_Select.py` L231 | `self.background_image.copy()` + `set_alpha()` **cada frame** | Surface de background no pre-cacheada con alpha |
| **MEDIA** | `game/hub/screen_controller.py` | **Recrea instancias de pantallas** completas en cada navegación | Sin cache de pantallas |
| **BAJA** | `game/ui/pause_menu.py` L14-15 | Crea fuentes en cada `draw()` | Sin cache |
| **BAJA** | `game/pages/home.py` L265 | `draw_climate_warning()` crea fuente cada frame | Sin cache |
| **DISEÑO** | `game/components/Button.py` | `draw()` mezcla rendering con lógica de input | Anti-patrón SRP |
| **DISEÑO** | `game/components/PopUp.py` L174-228 | Lógica de eventos duplicada | `MOUSEMOTION` y `MOUSEBUTTONDOWN` procesados dos veces |

### Acciones a implementar

| # | Tarea | Archivo(s) | Detalle |
|---|-------|-----------|---------|
| 5.1 | Cachear surfaces de partículas | `game/components/EnvironmentalEffects.py` | Pre-crear las surfaces en `__init__` y reutilizarlas con `fill()` en vez de crear nuevas cada frame. Usar `pygame.Surface.convert_alpha()` |
| 5.2 | Separar input de rendering en ProfileManager | `game/pages/home.py` + `game/pages/profile_manager.py` | Extraer los rects de UI a un dict cacheado (`self._ui_rects`) que se actualice solo en el draw, no re-dibujar para obtener rects |
| 5.3 | Cache de fuentes global | Nuevo: `game/components/FontCache.py` | Crear singleton `FontCache` con `get_font(name, size)` → cachea por `(name, size)`. Usar en `Card.py`, `PauseMenu`, `home.py`, `PopUp.py` |
| 5.4 | Eliminar `time.sleep()` bloqueante | `game/components/AudioManager.py` | Usar `pygame.mixer.music.fadeout(500)` con callback no-bloqueante en vez de `time.sleep(0.5)` |
| 5.5 | Pre-cachear background con alpha | `game/pages/Level_Select.py` | En `__init__`: crear `self._bg_with_alpha` = background con alpha aplicada una sola vez |
| 5.6 | Cache de pantallas en ScreenController | `game/hub/screen_controller.py` | Cachear las instancias de `HockeyMainScreen` y `LevelSelectScreen`, solo recrear si hay cambio de perfil/save |
| 5.7 | Cachear fuentes en componentes | `Card.py`, `PauseMenu.py`, `home.py` | Reemplazar `pygame.font.Font(None, size)` en draw() por `self._font_cache[size]` |
| 5.8 | Fix doble procesamiento de PopUp | `game/components/PopUp.py` | Eliminar la lógica duplicada en `handle_event()` L174-228 |

### Skills / Herramientas
- **Skill:** `python-design-patterns` → Singleton para `FontCache`, SRP para separar input/rendering
- **Skill:** `pygame-patterns` → patrones de cache de surfaces, `convert_alpha()`
- **MCP:** `mcp_context7_get-library-docs` → documentación de `pygame.Surface.convert()` vs `convert_alpha()`, `pygame.font.Font` caching
- **Subagente especializado:** Investigar en internet las mejores prácticas de rendimiento en pygame:
  - `Surface.convert()` y `Surface.convert_alpha()` para blitting rápido
  - Dirty rect rendering con `pygame.sprite.RenderUpdates`
  - `pygame.font.Font` caching
  - `pygame.Surface` pooling

### Criterio de aceptación
- [ ] Selección de skins es fluida (>30 FPS constantes)
- [ ] Navegación entre pantallas sin lag perceptible
- [ ] No se crean surfaces nuevas cada frame en EnvironmentalEffects
- [ ] No se crean fuentes nuevas cada frame en ningún componente
- [ ] No hay `time.sleep()` en thread principal

---

## Mejora 6: Componente Button universal optimizado {#mejora-6}

### Problema actual
El `Button` actual (`game/components/Button.py`) es un botón **circular basado en sprite PNG** que:
1. Mezcla rendering con lógica de input en `draw()`
2. Solo soporta imágenes PNG (no texto, colores, bordes)
3. `is_clicked()` usa `get_pressed()[0]` (continuo, no discreto)
4. Re-escala imagen en cada transición hover

### Diseño del nuevo componente

```python
class GameButton:
    """Botón universal optimizado para toda la UI del juego.
    
    Soporta:
    - Texto con fuente, color y tamaño configurables
    - Background de color sólido o imagen PNG
    - Bordes con radio, color y grosor configurables
    - Animaciones de hover (escala, color, glow)
    - Estados: normal, hover, pressed, disabled
    - Sonido de hover/click vía AudioManager
    """
```

### Acciones a implementar

| # | Tarea | Archivo(s) | Detalle |
|---|-------|-----------|---------|
| 6.1 | Crear nuevo `GameButton` | Nuevo: `game/components/GameButton.py` | Clase con parámetros: `text`, `font_size`, `text_color`, `bg_color`, `hover_color`, `pressed_color`, `disabled_color`, `border_radius`, `border_color`, `border_width`, `bg_image`, `size`, `position`, `on_click`, `on_hover`, `padding`, `icon` |
| 6.2 | Implementar estados y transiciones | `game/components/GameButton.py` | Enum: `NORMAL`, `HOVER`, `PRESSED`, `DISABLED`. Transiciones suaves (lerp de color en 0.15s). Pre-renderizar surfaces por estado |
| 6.3 | Separar `update()` y `draw()` | `game/components/GameButton.py` | `update(events)` → procesa input, cambia estado. `draw(surface)` → solo renderiza. `is_clicked` basado en `MOUSEBUTTONDOWN` event, no polling |
| 6.4 | Pre-renderizar surfaces por estado | `game/components/GameButton.py` | Cache: `_surfaces = {NORMAL: Surface, HOVER: Surface, ...}`. Invalidar cache solo cuando cambian propiedades. Usar `convert_alpha()` |
| 6.5 | Soporte de animación de hover | `game/components/GameButton.py` | Scale suave 1.0 → 1.05 en 0.1s. Glow overlay con alpha gradient. Sin crear nuevas surfaces — usar transform cacheado |
| 6.6 | Integrar AudioManager | `game/components/GameButton.py` | `audio_manager.play_sound_effect("button_hover")` al entrar en hover, `play_sound_effect("button_click")` al click |
| 6.7 | Migrar Home a GameButton | `game/pages/home.py` | Reemplazar todos los `Button(...)` por `GameButton(...)`. Mantener retrocompatibilidad con botones circulares via `bg_image` |
| 6.8 | Migrar Level_Select a GameButton | `game/pages/Level_Select.py` | Reemplazar botones de "Volver" y "Jugar" |
| 6.9 | Migrar PvP Setup a GameButton | `game/pages/pvp_setup.py` | Reemplazar botones ◀▶, back, play |
| 6.10 | Migrar PopUp a GameButton | `game/components/PopUp.py` | Reemplazar botones internos del popup |
| 6.11 | Mantener Button antiguo como deprecated | `game/components/Button.py` | Agregar docstring `@deprecated`, no eliminarlo aún |

### Especificación técnica del `GameButton`

```python
# Constructor
GameButton(
    # Contenido
    text: str = "",
    icon: pygame.Surface | None = None,
    bg_image: pygame.Surface | None = None,
    
    # Geometría
    position: tuple[int, int] = (0, 0),
    size: tuple[int, int] | None = None,  # Auto-calculado si None
    padding: tuple[int, int] = (20, 10),
    border_radius: int = 8,
    
    # Colores por estado
    bg_color: tuple = (60, 60, 80),
    hover_color: tuple = (80, 80, 110),
    pressed_color: tuple = (40, 40, 60),
    disabled_color: tuple = (50, 50, 50),
    text_color: tuple = (255, 255, 255),
    border_color: tuple | None = None,
    border_width: int = 0,
    
    # Tipografía
    font_name: str | None = None,
    font_size: int = 24,
    
    # Animación
    hover_scale: float = 1.05,
    animation_speed: float = 0.15,  # segundos
    
    # Callbacks
    on_click: Callable | None = None,
    enabled: bool = True,
)

# Métodos públicos
def update(self, events: list[pygame.event.Event], dt: float) -> bool:
    """Procesa eventos. Retorna True si fue clickeado."""

def draw(self, surface: pygame.Surface) -> None:
    """Renderiza el botón en su estado actual."""

def set_text(self, text: str) -> None:
    """Actualiza texto e invalida cache."""

def set_enabled(self, enabled: bool) -> None:
    """Habilita/deshabilita el botón."""

@property
def rect(self) -> pygame.Rect:
    """Rect del botón para layout externo."""
```

### Skills / Herramientas
- **Skill:** `game-ui-design` → `references/patterns.md` para estándares de botones en juegos
- **Skill:** `python-design-patterns` → SRP (separar update/draw), composición
- **Skill:** `pygame-patterns` → cache de surfaces, input discreto via events
- **MCP:** `mcp_context7_get-library-docs` → `pygame.draw.rect` con `border_radius`, `pygame.transform.smoothscale`, `pygame.event.Event`
- **Subagente:** Investigar en internet implementaciones de botón optimizado en pygame:
  - `pygame-gui` library como referencia de API
  - Patrón Observer para callbacks
  - Dirty sprite pattern de pygame

### Criterio de aceptación
- [ ] `GameButton` soporta texto, colores, bordes, e imágenes
- [ ] `update()` y `draw()` separados (SRP)
- [ ] Surfaces pre-renderizados por estado (sin crear en cada frame)
- [ ] Animaciones de hover suaves
- [ ] Retrocompatible con botones circulares existentes (via `bg_image`)
- [ ] Todos los screens migrados al nuevo botón
- [ ] FPS estable sin drops al interactuar con botones

---

## Mejora 7: Rúbrica de rewards — golpe hacia portería + defensa {#mejora-7}

### Problema actual
Se necesitan dos nuevos rewards:
1. **Reward ofensivo:** Puntuación al agente cuando golpea el puck en dirección de la portería enemiga
2. **Reward defensivo:** Penalización/recompensa por comportamiento defensivo (evitar que le anoten)

### Diagnóstico del sistema de rewards actual
**Archivo:** `training/envs/base_env.py` → `_calculate_reward()` (línea ~156)

| Tipo | Reward actual | Descripción |
|------|--------------|-------------|
| Ofensivo | `+0.8` hit base | Golpear el puck |
| Ofensivo | `+2.5 * alignment * speed` | Shot Quality (ya existe parcialmente) |
| Ofensivo | `+0.5` | Hard Shot (speed > 0.6) |
| Ofensivo | `+5.0` | Gol anotado |
| Defensivo | `-3.0` | Gol concedido |
| Defensivo | `+1.0` | Interception |
| Defensivo | `-0.05` | Net-Front Discipline |
| Mixto | `+0.5` | Counterattack |

**Análisis:** El Shot Quality (`+2.5 * alignment * speed`) ya recompensa parcialmente el golpe hacia la portería, pero:
- `alignment` se calcula como ángulo entre vector de velocidad del puck y vector hacia portería rival — **esto es correcto pero insuficiente**
- No penaliza golpes que alejan el puck de la portería rival
- La defensa está poco incentivada: solo `-3.0` por gol concedido y `+1.0` por intercepción

### Acciones a implementar

| # | Tarea | Archivo(s) | Detalle |
|---|-------|-----------|---------|
| 7.1 | Mejorar reward de golpe hacia portería | `training/envs/base_env.py` → `_calculate_reward()` | Calcular `dot_product(puck_velocity, direction_to_enemy_goal)` post-golpe. Si > 0.7: `+1.5 * dot_product`. Si < 0: `-0.5` (penalizar golpe en propia dirección) |
| 7.2 | Agregar reward de posicionamiento defensivo | `training/envs/base_env.py` | `+0.1` cuando puck está en mitad IA y agente está entre puck y portería propia (blocking position). `+0.05` por mantener distancia < 100px del puck en mitad defensiva |
| 7.3 | Penalización por gol concedido con contexto | `training/envs/base_env.py` | Aumentar `-3.0` a `-4.0`. Añadir `-1.0` extra si la IA estaba lejos de la portería cuando fue gol (distancia > 60% del campo) = `-5.0` total por negligencia |
| 7.4 | Reward por despeje exitoso | `training/envs/base_env.py` | Cuando puck venía hacia portería IA (velocidad_x > 0) y post-golpe cambia a velocidad_x < 0: `+1.5` (despeje) |
| 7.5 | Reward por presión en portería rival | `training/envs/base_env.py` | Si puck está en mitad rival y yendo hacia portería rival: `+0.03/frame` |
| 7.6 | Añadir tracking de `last_puck_velocity_pre_hit` | `training/envs/base_env.py` | Guardar `puck.velocity` antes de cada golpe para comparar dirección pre/post golpe |
| 7.7 | Documentar rúbrica actualizada | Nuevo: `docs/reward_rubric.md` | Tabla completa con todos los rewards, sus valores, condiciones y justificación |
| 7.8 | Validar balance de rewards | Testing manual | Entrenar 100K steps con nueva rúbrica, comparar mean_reward y comportamiento visual vs rúbrica anterior |

### Rúbrica propuesta completa

```
┌──────────────────────────────────────────────────────────────────┐
│                    RÚBRICA DE REWARDS v2                         │
├──────────────────────┬──────────┬────────────────────────────────┤
│ Componente           │ Valor    │ Condición                      │
├──────────────────────┼──────────┼────────────────────────────────┤
│ OFENSIVOS                                                        │
├──────────────────────┼──────────┼────────────────────────────────┤
│ Hit base             │ +0.8     │ Golpear el puck                │
│ Shot to goal (NUEVO) │ +1.5*dp  │ dot_product(vel, to_goal) > 0.7│
│ Shot away (NUEVO)    │ -0.5     │ dot_product(vel, to_goal) < 0  │
│ Shot Quality         │ +2.5*a*s │ alignment * speed_ratio        │
│ Hard Shot            │ +0.5     │ speed_ratio > 0.6              │
│ Gol anotado          │ +5.0     │ AI marca gol                   │
│ Goal Pressure (NUEVO)│ +0.03/f  │ Puck en mitad rival + yendo→   │
│ Counterattack        │ +0.5     │ Ofensiva inmediata post-defensa│
├──────────────────────┼──────────┼────────────────────────────────┤
│ DEFENSIVOS                                                       │
├──────────────────────┼──────────┼────────────────────────────────┤
│ Interception         │ +1.0     │ Detiene puck hacia portería IA │
│ Clear (NUEVO)        │ +1.5     │ Despeje (invierte dirección X)│
│ Block Position(NUEVO)│ +0.1     │ Entre puck y portería en mitad │
│ Def. Proximity(NUEVO)│ +0.05    │ < 100px del puck en mitad def. │
│ Gol concedido        │ -4.0     │ Rival marca (antes era -3.0)   │
│ Negligencia (NUEVO)  │ -1.0     │ Lejos de portería al conceder  │
│ Net-Front Discipline │ -0.05    │ Demasiado cerca de su portería │
├──────────────────────┼──────────┼────────────────────────────────┤
│ POSICIONALES                                                     │
├──────────────────────┼──────────┼────────────────────────────────┤
│ Gap Control          │ +0.05    │ ~100px del puck en mitad IA    │
│ Approach             │ +0.08*f  │ Reducir distancia cuando >150px│
│ Positional Play      │ +0.03    │ Entre puck y portería propia   │
│ Y-Axis Alignment     │ +0.02    │ Alineación vertical con puck   │
│ Pressure Play        │ +0.01    │ Cerca del centro sin puck      │
│ Inactivity           │ -0.02    │ Quieto con puck cerca          │
└──────────────────────┴──────────┴────────────────────────────────┘
```

### Skills / Herramientas
- **Skill:** `stable-baselines3` → reward shaping best practices, `references/algorithms.md`
- **Skill:** `pygame-patterns` → cálculo de vectores y dot product
- **Subagente especializado:** Investigar en internet:
  - Reward shaping para juegos adversariales (papers: "Reward Shaping in RL for competitive games")
  - Balance ofensivo/defensivo en RL para juegos tipo Air Hockey / Pong
  - Potencial-based reward shaping (PBRS) para evitar degenerate policies
  - "Reward hacking" en juegos adversariales y cómo prevenirlo
- **MCP:** `mcp_context7_get-library-docs` → stable-baselines3 reward customization
- **Archivo existente:** `shared/physics.py` → `dot_product()`, `calculate_vector()`, `normalize_vector()` — funciones ya disponibles para cálculos vectoriales

### Criterio de aceptación
- [ ] El agente recibe reward por golpear puck hacia portería rival
- [ ] El agente recibe penalización por golpear puck en dirección incorrecta
- [ ] El agente recibe reward por posicionamiento defensivo
- [ ] El agente recibe penalización extra por negligencia defensiva
- [ ] El agente recibe reward por despejes exitosos
- [ ] Mean reward en evaluación igual o superior al modelo anterior
- [ ] Comportamiento visual muestra más juego posicional defensivo

---

## Mejora 8: Oponente de entrenamiento mejorado {#mejora-8}

### Problema actual
El oponente simulado en `training/envs/base_env.py` → `_update_human_player()` es una heurística simple que:
- Predice posición Y del puck con error gaussiano
- Persigue con `reaction_speed` variable
- Progresión: `opponent_skill` sube de 0.3 a 0.9 en steps de 0.1

Esto puede ser insuficiente para entrenar un agente robusto.

### Análisis: Algorítmico vs Redes Adversarias

| Criterio | Oponente Algorítmico | GAN/Self-Play |
|----------|---------------------|---------------|
| **Complejidad de implementación** | Baja | Alta |
| **Estabilidad de entrenamiento** | Alta | Media-Baja (modo colapso) |
| **Diversidad de comportamiento** | Media (determinístico con ruido) | Alta (adapativo) |
| **Costo computacional** | Bajo | +100% (dos redes) |
| **Capacidad de currículo** | Alta (parámetros directos) | Media (requiere matchmaking) |
| **Tiempo de convergencia** | Rápido | Lento (equilibrio Nash) |
| **Recomendación** | ✅ **Para este proyecto** | Para proyectos más grandes |

**Recomendación:** Mejorar el oponente algorítmico con comportamientos más sofisticados. El self-play tiene riesgos de inestabilidad y complejidad desproporcionada para un Air Hockey 2D.

### Acciones a implementar

| # | Tarea | Archivo(s) | Detalle |
|---|-------|-----------|---------|
| 8.1 | Investigar oponentes algorítmicos para Air Hockey | Internet/papers | Buscar papers y repositorios sobre oponentes algorítmicos para Pong/Air Hockey/table games |
| 8.2 | Crear oponente algorítmico con 3 modos | Nuevo: `training/envs/opponents.py` | Modo defensivo (posicionarse entre puck y portería), modo ofensivo (atacar hacia portería rival), modo predictivo (interceptar trayectoria futura del puck) |
| 8.3 | Implementar selección de comportamiento por estado | `training/envs/opponents.py` | Si puck viene hacia él → **defensivo**: moverse a posición de bloqueo. Si puck en su mitad → **ofensivo**: avanzar y golpear hacia portería rival. Si puck lejos → **posicional**: mantener posición central |
| 8.4 | Implementar golpe direccional inteligente | `training/envs/opponents.py` | Al golpear: calcular ángulo hacia portería rival con offset aleatorio basado en `accuracy`. Aplicar velocidad al puck proporcional a `aggression` |
| 8.5 | Agresividad variable por skill level | `training/envs/opponents.py` | Skill baja: mayormente defiende, errores frecuentes. Skill media: alterna defensa/ataque, errores moderados. Skill alta: ataque controlado, errores mínimos |
| 8.6 | Variabilidad de comportamiento (anti-sobreajuste) | `training/envs/opponents.py` | Cada episodio: randomizar estilo (60% algorítmico, 20% agresivo, 20% defensivo). Añadir ruido en decisiones proporcional a `1 - skill` |
| 8.7 | Mejorar sistema de currículo | `training/configs/curriculum.py` | Agregar métricas más granulares: win_rate, average_goals_scored, average_goals_conceded, rally_length. Subir dificultad solo si win_rate > 65% **Y** avg_goals_scored > 3 |
| 8.8 | Integrar en el entorno | `training/envs/base_env.py` | Reemplazar `_update_human_player()` por `self.opponent.update(puck, ai_mallet)` del nuevo sistema modular |
| 8.9 | Añadir opción de self-play futuro | `training/envs/opponents.py` | Crear clase `SelfPlayOpponent` que cargue un modelo PPO previo como oponente. No implementar ahora, solo la interface |
| 8.10 | Tests de validación | `tests/test_training.py` | Verificar que el oponente: (a) no se queda quieto, (b) defiende cuando puck viene, (c) ataca cuando tiene el puck, (d) varía comportamiento entre episodios |

### Diseño del oponente algorítmico mejorado

```python
class AlgorithmicOpponent:
    """Oponente algorítmico con comportamiento basado en estados."""
    
    class State(Enum):
        DEFENSIVE = "defensive"    # Puck viene hacia él
        OFFENSIVE = "offensive"    # Puck en su mitad, puede atacar
        POSITIONAL = "positional"  # Puck lejos, mantener posición
        INTERCEPT = "intercept"    # Puck predecible, interceptar
    
    def __init__(self, skill: float = 0.3, style: str = "balanced"):
        self.skill = skill           # 0.0 - 1.0
        self.style = style           # "balanced", "aggressive", "defensive"
        self.reaction_delay = ...    # Frames de delay antes de reaccionar
        self.prediction_error = ...  # Error gaussiano en predicción
        self.shot_accuracy = ...     # Precisión del golpe
    
    def update(self, puck_pos, puck_vel, own_pos, field) -> tuple[float, float]:
        state = self._determine_state(puck_pos, puck_vel)
        if state == State.DEFENSIVE:
            target = self._calculate_block_position(puck_pos, puck_vel, own_goal)
        elif state == State.OFFENSIVE:
            target = self._calculate_attack_position(puck_pos, enemy_goal)
        elif state == State.INTERCEPT:
            target = self._predict_intercept_point(puck_pos, puck_vel)
        else:
            target = self._get_home_position()
        
        # Movimiento con delay y error
        return self._move_towards(target, noise=1-self.skill)
```

### Skills / Herramientas
- **Skill:** `stable-baselines3` → curriculum learning, opponent design
- **Skill:** `game-developer` → game AI patterns (state machines para oponentes)
- **Subagente especializado:** Investigar en internet:
  - "Algorithmic opponents for reinforcement learning training"
  - "Curriculum learning in competitive games"
  - "OpenAI five opponent pool" (para referencias de self-play)
  - "Pong AI opponent reward shaping"
  - "Procedural opponent difficulty scaling"
- **MCP:** `mcp_context7_get-library-docs` → stable-baselines3 VecEnv opponent wrapping

### Criterio de aceptación
- [ ] Oponente tiene al menos 3 comportamientos: defensivo, ofensivo, posicional
- [ ] Oponente predice trayectoria del puck para interceptar
- [ ] Oponente golpea el puck en dirección de la portería rival
- [ ] Dificultad escala suavemente de novato a experto
- [ ] Variabilidad entre episodios para evitar sobreajuste
- [ ] El agente entrenado contra este oponente juega mejor que con el oponente anterior

---

## Mejora 9: Optimización de arquitectura del agente RL {#mejora-9}

### Estado actual de la arquitectura

| Parámetro | v2_quick | v2_standard | v2_deep |
|-----------|----------|-------------|---------|
| **Timesteps** | 800K | 3M | 5M |
| **Policy Net (pi)** | [256, 128] | [512, 256, 128] | [512, 256, 128] |
| **Value Net (vf)** | [256, 128] | [512, 256, 128] | [512, 256, 128] |
| **Learning Rate** | 3e-4 → 5e-6 | 3e-4 → 1e-5 | 2e-4 → 5e-6 |
| **Batch Size** | 128 | 256 | 256 |
| **Gamma** | 0.995 | 0.997 | 0.997 |
| **Entropy Coef** | 0.015 | 0.015 | 0.01 |
| **Clip Range** | 0.2 | 0.15 | 0.12 |
| **Obs Space** | 13D | 13D | 13D |
| **Action Space** | Discrete(9) | Discrete(9) | Discrete(9) |
| **Activación** | ReLU | ReLU | ReLU |
| **Algoritmo** | PPO | PPO | PPO |

**Best model actual:** v2_quick con `best_mean_reward = 209.14` (800K steps, 55 min training)

### Acciones a implementar

| # | Tarea | Archivo(s) | Detalle |
|---|-------|-----------|---------|
| 9.1 | Investigar hiperparámetros óptimos para PPO | Internet | Consultar: "PPO hyperparameter tuning for discrete action games", Optuna/Ray Tune para búsqueda automática |
| 9.2 | Evaluar observaciones adicionales | `training/envs/base_env.py` | Añadir al obs space: (a) velocidad del mallet IA (2D), (b) velocidad del oponente (2D), (c) distancia a cada portería. Total: 13 → 17-19D |
| 9.3 | Evaluar acciones continuas vs discretas | `training/envs/base_env.py` | Experimentar con `Box(2)` (dx, dy continuos) en vez de `Discrete(9)` — permite movimiento más fino. PPO soporta ambos |
| 9.4 | Optimizar arquitectura de red | `training/configs/ppo_configs.py` | Crear preset `v3_optimized`: `pi=[256, 256]`, `vf=[128, 128]` (red más ancha para pi, más estrecha para vf). Probar `Tanh` vs `ReLU` en capas internas |
| 9.5 | Agregar normalización de observaciones | `training/train.py` | Envolver env en `VecNormalize` de SB3 para normalizar obs y rewards automáticamente |
| 9.6 | Frame stacking | `training/train.py` | Envolver env en `VecFrameStack(n_stack=4)` para dar contexto temporal al agente (ve 4 frames consecutivos = 13*4=52D) |
| 9.7 | Implement n_steps tuning | `training/configs/ppo_configs.py` | Probar `n_steps=2048` (default) vs `n_steps=4096` para rollouts más largos. Para juegos adversariales, rollouts más largos capturan mejor las dinámicas |
| 9.8 | Learning rate warmup | `training/configs/ppo_configs.py` | Implementar warmup + cosine decay: LR empieza bajo (1e-5), sube a 3e-4 en 10% del training, luego cosine decay |
| 9.9 | GAE Lambda tuning | `training/configs/ppo_configs.py` | Default `gae_lambda=0.95`. Probar `0.98` para mejor estimación de ventaja en juegos con rewards espaciados |
| 9.10 | Evaluar DQN vs PPO | Análisis/investigación | Para Discrete(9), DQN puede ser más sample-efficient. Crear config DQN para comparar: `DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=100000, double_q=True, dueling=True, prioritized_replay=True)` |
| 9.11 | Comparar modelos | `training/analysis/model_comparison.py` | Entrenar 5 variantes (100K steps c/u), comparar con `compare_models()`. Métricas: mean_reward, std_reward, win_rate, avg_goals |
| 9.12 | Documentar resultados | `docs/model_architecture_analysis.md` | Tabla comparativa de todas las variantes probadas |

### Investigaciones específicas a realizar

#### 9.A — Búsqueda de hiperparámetros (Subagente)
```
Investigar en internet:
1. "PPO hyperparameters for Atari/board games" (papers de OpenAI, DeepMind)
2. "Optuna SB3 integration" para búsqueda automática
3. "Rainbow DQN vs PPO for discrete action spaces"
4. "Dueling DQN for competitive games"
5. SB3 Zoo hyperparameters: github.com/DLR-RM/rl-baselines3-zoo
```

#### 9.B — Mejoras de observación (Subagente)
```
Investigar en internet:
1. "Feature engineering for RL game agents"
2. "Observation space design for competitive games"
3. "CNN vs MLP for simple game observations"
4. "Relative vs absolute coordinates in RL"
```

#### 9.C — Técnicas avanzadas de entrenamiento (Subagente)
```
Investigar en internet:
1. "Proximal Policy Optimization tricks" (PPO implementation details blog)
2. "Reward normalization in SB3"
3. "Generalized Advantage Estimation (GAE) tuning"
4. "PPO clip range scheduling"
5. "MiniBatch normalization for RL"
```

### Preset propuesto v3_optimized

```python
"v3_optimized": {
    "total_timesteps": 2_000_000,
    "learning_rate": cosine_warmup_schedule(peak=3e-4, warmup_frac=0.1),
    "n_steps": 4096,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.997,
    "gae_lambda": 0.98,
    "clip_range": linear_schedule(0.2, 0.05),
    "ent_coef": linear_schedule(0.02, 0.005),  # Más exploración al inicio
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.025,
    "policy_kwargs": {
        "net_arch": {
            "pi": [256, 256],
            "vf": [128, 128],
        },
        "activation_fn": torch.nn.ReLU,
        "ortho_init": True,
    },
    "normalize_advantage": True,
}
```

### Skills / Herramientas
- **Skill:** `stable-baselines3` → algoritmos, hiperparámetros, VecNormalize, VecFrameStack
- **Skill:** `pytorch` → arquitectura de red, activaciones, optimizadores
- **MCP:** `mcp_context7_get-library-docs` → stable-baselines3 PPO, DQN, VecNormalize docs
- **Subagentes especializados:** 3 subagentes de investigación (ver 9.A, 9.B, 9.C arriba)
- **Terminal:** `python -m training.train --preset v3_optimized --n-envs 4` para entrenar

### Criterio de aceptación
- [ ] Al menos 3 variantes de arquitectura evaluadas
- [ ] Comparación documentada con métricas cuantitativas
- [ ] Nueva configuración v3_optimized con justificación de cada hiperparámetro
- [ ] Mean reward > 209.14 (superar modelo actual)
- [ ] Documento de análisis con gráficas de entrenamiento

---

## Mejora 10: Componentizar UI para evitar re-renderizados {#mejora-10}

### Problema actual
La UI re-renderiza todo cada frame: backgrounds, títulos, elementos estáticos. Esto consume recursos innecesariamente, especialmente combinado con los problemas de la Mejora 5.

### Diagnóstico técnico
En pygame, la optimización de renderizado se basa en:

1. **Dirty Rect Rendering:** Solo redibujar las áreas que cambiaron
2. **Surface Caching:** Pre-renderizar elementos estáticos en surfaces y blitear
3. **Layer System:** Separar capas estáticas (background, títulos) de dinámicas (botones hover, partículas)
4. **`convert()` / `convert_alpha()`:** Convertir surfaces al formato de display para blitting rápido

### Acciones a implementar

| # | Tarea | Archivo(s) | Detalle |
|---|-------|-----------|---------|
| 10.1 | Investigar pygame rendering optimization | Internet/MCP | Consultar: `pygame.display.update(rect_list)` vs `flip()`, dirty sprites, `RenderUpdates` group |
| 10.2 | Crear sistema de capas (LayerManager) | Nuevo: `game/components/LayerManager.py` | Clase que gestiona 3 capas: `STATIC` (background, marcos, títulos), `DYNAMIC` (botones, paneles interactivos), `OVERLAY` (popups, tooltips, partículas) |
| 10.3 | Implementar capa estática cacheada | `game/components/LayerManager.py` | La capa `STATIC` renderiza una sola vez a un `Surface` y se re-blitea cada frame sin recalcular. Se invalida manualmente con `invalidate_static()` |
| 10.4 | Migrar Home screen a LayerManager | `game/pages/home.py` | Mover background + título + marco de panels a capa STATIC. Botones y ProfileManager a capa DYNAMIC |
| 10.5 | Migrar Level Select a LayerManager | `game/pages/Level_Select.py` | Background + grid cards (solo cambian en selección) como STATIC. Level details + botones como DYNAMIC |
| 10.6 | Migrar PvP Setup a LayerManager | `game/pages/pvp_setup.py` | Layout base como STATIC. Selectores y botones como DYNAMIC |
| 10.7 | Aplicar `convert()` / `convert_alpha()` | Todos los archivos que cargan surfaces | Asegurar que toda surface cargada con `pygame.image.load()` use `.convert()` (sin alpha) o `.convert_alpha()` (con alpha) inmediatamente |
| 10.8 | Implementar dirty rect para botones | `game/components/GameButton.py` | Cuando el botón cambia de estado, reportar su rect como "dirty" al LayerManager. Solo redibujar esa área |
| 10.9 | Optimizar EnvironmentalEffects | `game/components/EnvironmentalEffects.py` | Pre-crear pool de surfaces para partículas en `__init__` (con `convert_alpha()`). En `update()`: modificar surfaces existentes. Nunca crear surfaces nuevas en el loop |
| 10.10 | Font rendering cache | `game/components/FontCache.py` | Para textos estáticos (títulos, labels), cachear `font.render()` result por `(text, color, size)`. Invalidar solo si el texto cambia |
| 10.11 | Profiling de rendimiento | Terminal | Usar `cProfile` o `py-spy` para medir FPS antes y después. Target: >60 FPS en todas las pantallas |

### Diseño del LayerManager

```python
class LayerManager:
    """Gestiona capas de rendering para evitar re-renderizados innecesarios."""
    
    class Layer(Enum):
        STATIC = 0      # Background, títulos, marcos
        DYNAMIC = 1     # Botones, paneles interactivos
        OVERLAY = 2     # Popups, tooltips, partículas
    
    def __init__(self, screen_size: tuple[int, int]):
        self._surfaces = {
            Layer.STATIC: pygame.Surface(screen_size).convert(),
            Layer.DYNAMIC: pygame.Surface(screen_size, pygame.SRCALPHA).convert_alpha(),
            Layer.OVERLAY: pygame.Surface(screen_size, pygame.SRCALPHA).convert_alpha(),
        }
        self._static_valid = False
        self._dirty_rects: list[pygame.Rect] = []
    
    def begin_static(self) -> pygame.Surface:
        """Retorna surface estática para dibujar (solo si invalidada)."""
        if self._static_valid:
            return None  # No necesita redibujar
        return self._surfaces[Layer.STATIC]
    
    def end_static(self):
        """Marca la capa estática como válida."""
        self._static_valid = True
    
    def invalidate_static(self):
        """Fuerza re-renderizado de capa estática (ej: cambio de pantalla)."""
        self._static_valid = False
    
    def get_dynamic(self) -> pygame.Surface:
        """Retorna surface dinámica (se limpia cada frame)."""
        self._surfaces[Layer.DYNAMIC].fill((0, 0, 0, 0))
        return self._surfaces[Layer.DYNAMIC]
    
    def get_overlay(self) -> pygame.Surface:
        """Retorna surface de overlay (se limpia cada frame)."""
        self._surfaces[Layer.OVERLAY].fill((0, 0, 0, 0))
        return self._surfaces[Layer.OVERLAY]
    
    def compose(self, screen: pygame.Surface):
        """Compone todas las capas en la pantalla final."""
        screen.blit(self._surfaces[Layer.STATIC], (0, 0))
        screen.blit(self._surfaces[Layer.DYNAMIC], (0, 0))
        screen.blit(self._surfaces[Layer.OVERLAY], (0, 0))
    
    def add_dirty_rect(self, rect: pygame.Rect):
        """Marca un rect como sucio para dirty rect optimization."""
        self._dirty_rects.append(rect)
```

### Skills / Herramientas
- **Skill:** `pygame-patterns` → Surface caching, `convert()` / `convert_alpha()`, dirty rect rendering
- **Skill:** `python-design-patterns` → Separation of Concerns (rendering layers), composition
- **Skill:** `game-developer` → rendering pipelines, draw call optimization
- **MCP:** `mcp_context7_get-library-docs` → documentación de pygame:
  - `pygame.display.update()` vs `pygame.display.flip()`
  - `pygame.sprite.RenderUpdates` y `pygame.sprite.LayeredUpdates`
  - `pygame.Surface.convert()` rendimiento
  - `pygame.Surface.subsurface()` para regiones
- **Subagente especializado:** Investigar en internet:
  - "Pygame rendering optimization techniques 2024"
  - "Dirty rect rendering pygame"
  - "Pygame menu system optimization"
  - "pygame-gui source code" (referencia de componentes optimizados)
  - "Pygame surface pooling pattern"

### Criterio de aceptación
- [ ] Background no se re-renderiza cada frame
- [ ] Títulos y textos estáticos se cachean
- [ ] Todas las surfaces usan `convert()` o `convert_alpha()`
- [ ] FPS > 60 en pantalla Home con todos los efectos activos
- [ ] FPS > 60 en Level Select con scroll
- [ ] Profiling confirma reducción > 50% de tiempo de rendering
- [ ] Sin parpadeo ni artefactos visuales

---

## Orden de ejecución recomendado

Las mejoras tienen dependencias entre sí. El orden óptimo de implementación es:

```
Fase 1 — Fundamentos (prerequisitos para todo lo demás)
├── Mejora 6: GameButton universal          ← base para toda la UI
├── Mejora 10: LayerManager + componentizar ← rendimiento base
└── Mejora 5: Fix performance específicos   ← depende de 6 y 10

Fase 2 — UI/UX funcional
├── Mejora 1: Botones Home                  ← usa GameButton (Mejora 6)
├── Mejora 2: Pantalla PvP Setup            ← usa GameButton + LayerManager
├── Mejora 3: Porterías por nivel           ← independiente, rápido
└── Mejora 4: Cronómetro visible            ← independiente, rápido

Fase 3 — IA y entrenamiento
├── Mejora 7: Rewards actualizados          ← prerequisito para 8 y 9
├── Mejora 8: Oponente mejorado             ← depende de rewards (Mejora 7)
└── Mejora 9: Arquitectura optimizada       ← depende de 7 y 8
```

### Estimación de esfuerzo

| Mejora | Complejidad | Archivos tocados | Archivos nuevos | Estimación |
|--------|-------------|-----------------|-----------------|------------|
| 1. Botones Home | Baja | 1 | 0 | 1-2 horas |
| 2. PvP Setup | Media | 2-3 | 0 | 3-4 horas |
| 3. Porterías | Baja | 1-2 | 0 | 30 min - 1 hora |
| 4. Cronómetro | Media | 3 | 0 | 2-3 horas |
| 5. Performance | Alta | 6-8 | 1 | 4-6 horas |
| 6. GameButton | Alta | 8-10 | 1 | 5-7 horas |
| 7. Rewards | Media | 2 | 1 | 2-3 horas |
| 8. Oponente | Alta | 3-4 | 1 | 4-6 horas |
| 9. Arquitectura RL | Alta | 3-4 | 1 | 6-10 horas (+ entrenamiento) |
| 10. LayerManager | Alta | 5-6 | 2 | 5-7 horas |
| **Total** | | **~40 archivos** | **~7 archivos** | **~33-49 horas** |

---

## Archivos nuevos a crear

| Archivo | Propósito | Mejora |
|---------|----------|--------|
| `game/components/GameButton.py` | Componente botón universal | 6 |
| `game/components/FontCache.py` | Cache singleton de fuentes | 5, 10 |
| `game/components/LayerManager.py` | Gestor de capas de rendering | 10 |
| `training/envs/opponents.py` | Oponente algorítmico mejorado | 8 |
| `docs/reward_rubric.md` | Documentación de rúbrica de rewards | 7 |
| `docs/model_architecture_analysis.md` | Análisis de arquitectura | 9 |
| `training/configs/v3_configs.py` | Configuración v3 optimizada (o agregar a `ppo_configs.py`) | 9 |

## Archivos a modificar

| Archivo | Mejoras que lo afectan |
|---------|----------------------|
| `game/pages/home.py` | 1, 5, 6, 10 |
| `game/pages/pvp_setup.py` | 2, 6, 10 |
| `game/pages/Level_Select.py` | 5, 6, 10 |
| `game/pages/profile_manager.py` | 5, 6 |
| `game/hub/screen_controller.py` | 2, 5 |
| `game/components/Button.py` | 6 (deprecated) |
| `game/components/EnvironmentalEffects.py` | 5, 10 |
| `game/components/Card.py` | 5, 10 |
| `game/components/PopUp.py` | 5, 6 |
| `game/ui/hud.py` | 4 |
| `game/ui/pause_menu.py` | 5 |
| `game/core/game_engine.py` | 3, 4 |
| `game/core/game_state.py` | 4 |
| `shared/utils/sprite_loader.py` | 3 |
| `training/envs/base_env.py` | 7, 8 |
| `training/configs/ppo_configs.py` | 9 |
| `training/configs/curriculum.py` | 8 |
| `training/train.py` | 9 |
