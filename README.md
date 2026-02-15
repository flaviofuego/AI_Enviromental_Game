# 🏒 Hockey is Melting Down

Juego de Air Hockey con temática medioambiental y agente IA entrenado mediante Reinforcement Learning (PPO / Stable-Baselines3).

## Características principales

### Juego

- 5 niveles temáticos con mecánicas únicas (UV Zones, Fog of War, Shrinking Field, Heat Waves)
- Sistema de power-ups con efectos sobre el puck y los mallets
- Modos Player vs IA y Player vs Player (teclado)
- HUD, pantalla de pausa (Continuar / Reiniciar / Menú), Game Over
- Assets y música por nivel, SpriteLoader centralizado
- Sistema de guardado de perfil y progreso

### IA

- **9 acciones** con movimiento diagonal (Up, Down, Left, Right, Stay, UpLeft, UpRight, DownLeft, DownRight)
- **Recompensas estilo hockey**: control de gap, interceptación, calidad de tiro, contraataque
- Observaciones de 13 dimensiones (base) o 26 dimensiones (con power-ups)
- Carga inteligente del mejor modelo (`metadata.json` → score + mtime)
- Correcciones comportamentales en tiempo real (`behavioral_corrections.py`)

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/yourusername/AI_Enviromental_Game.git
cd AI_Enviromental_Game

# Instalar dependencias (requiere uv)
make install
```

### Requisitos

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) como gestor de entorno

### Dependencias principales

- pygame, gymnasium, stable-baselines3 (con extras), torch, tensorboard
- numpy, matplotlib, pandas, seaborn

## Uso

### Jugar

```bash
make play            # Menú principal → selección de niveles
make play-quick      # Partida rápida (nivel 1, Player vs IA)
make play-level LEVEL=3   # Nivel específico
```

### Entrenar la IA

```bash
make train           # v2_standard (~500k steps)
make train-quick     # v2_quick (~100k steps, para pruebas)
make train-deep      # v2_deep (~2M steps)
make train-powerups  # Entrenar con power-ups (obs 26-dim)
make train-parallel  # 4 entornos simultáneos
make train-resume    # Reanudar desde último checkpoint
```

#### Parámetros personalizados

Todos los presets aceptan overrides por CLI:

```bash
uv run python -m training.train \
  --preset v2_standard --env base \
  --timesteps 300000 --lr 3e-4 --ent-coef 0.02 \
  --batch-size 128 --gamma 0.995 --n-envs 4
```

Opciones: `--timesteps`, `--epochs`, `--lr`, `--batch-size`, `--gamma`, `--ent-coef`, `--clip-range`, `--n-steps`, `--checkpoint-freq`, `--eval-freq`, `--n-envs`, `--name`, `--resume`.

### Monitorear entrenamiento

```bash
make tensorboard     # TensorBoard en http://localhost:6006
```

## Estructura del proyecto

```
├── game/                        # Código del juego
│   ├── main_hub.py              # Entry point: menú principal
│   ├── main.py                  # Entry point: partida rápida
│   ├── ai/                      # Sistema de IA
│   │   ├── model_loader.py      # Carga inteligente de modelos
│   │   ├── observation_builder.py  # Obs según tipo de modelo
│   │   └── behavioral_corrections.py
│   ├── core/                    # Motor del juego
│   │   ├── game_engine.py       # Loop principal, estados, render
│   │   ├── game_state.py        # FSM (MENU → PLAYING → PAUSED → GAME_OVER)
│   │   ├── match_manager.py     # Puntuación, timer, goles
│   │   ├── mechanics.py         # Mecánicas por nivel (UV, Fog, Shrink, Heat)
│   │   └── renderer.py         
│   ├── entities/                # Entidades del juego
│   │   ├── ai_mallet.py         # Mallet controlado por modelo RL
│   │   ├── human_mallet.py      # Mallet controlado por mouse
│   │   ├── keyboard_mallet.py   # Mallet controlado por teclado
│   │   └── powerups.py          # Power-ups y efectos
│   ├── config/                  # Configuración de niveles y guardado
│   │   ├── level_config.py      # Temas, mecánicas, assets por nivel
│   │   └── save_system.py       # Perfiles y progreso
│   ├── modes/                   # Modos de juego
│   │   ├── player_vs_ai.py
│   │   └── player_vs_player.py
│   ├── ui/                      # Interfaz
│   │   ├── hud.py               # Marcador, timer
│   │   ├── pause_menu.py        # Continuar / Reiniciar / Menú
│   │   └── game_over_screen.py
│   ├── components/              # Componentes UI reutilizables
│   │   ├── Button.py, Card.py, PopUp.py
│   │   └── AudioManager.py
│   ├── pages/                   # Páginas del hub
│   │   ├── home.py
│   │   └── Level_Select.py
│   └── assets/                  # Sprites, fondos, sonidos por nivel
│
├── shared/                      # Módulos compartidos (juego + training)
│   ├── config.py                # GameConfig, COLORS (sin dependencia de pygame)
│   ├── physics.py               # Colisiones y física
│   ├── entities/                # Entidades base (Mallet, Puck, Table)
│   └── utils/                   # SpriteLoader, drawing helpers
│
├── training/                    # Pipeline de entrenamiento RL
│   ├── train.py                 # CLI unificado (--preset, --env, overrides)
│   ├── envs/
│   │   ├── base_env.py          # Env 9-acciones, 13-dim obs, reward hockey
│   │   └── powerups_env.py      # Env con power-ups, 26-dim obs
│   ├── configs/
│   │   ├── ppo_configs.py       # Presets: v2_quick, v2_standard, v2_deep
│   │   └── curriculum.py        # Progresión de dificultad
│   ├── callbacks/               # Callbacks SB3
│   │   ├── behavior_analysis.py # Análisis de acciones (9 labels)
│   │   ├── movement_balance.py  # Balance de movimiento
│   │   └── difficulty_progression.py
│   └── analysis/                # Herramientas de análisis post-training
│
├── models/                      # Modelos entrenados (auto-generado)
│   └── <run_name>/
│       ├── best_model/          # Mejor checkpoint (model.zip)
│       ├── metadata.json        # Score, hiperparámetros, tiempo
│       └── *_final.zip          # Modelo final
│
├── logs/                        # Logs de TensorBoard (auto-generado)
├── Makefile                     # Comandos: play, train, tensorboard, clean
├── pyproject.toml               # Dependencias (uv)
└── README.md
```

## Niveles

| Nivel | Tema | Mecánica | Mensaje ambiental |
|-------|------|----------|--------------------|
| 1 | Arctic Meltdown | Tutorial — sin mecánica especial | Plástico en los océanos |
| 2 | Ozone Shield | **UV Zones** — zonas que aceleran el puck 30% | Gases CFC y capa de ozono |
| 3 | Smog Storm | **Fog of War** — visibilidad reducida | Contaminación del aire |
| 4 | Vanishing Forest | **Shrinking Field** — campo que se encoge | Deforestación |
| 5 | Urban Heat Island | **Heat Waves** — distorsión y menor fricción | Islas de calor urbanas |

## Licencia

Ver [LICENSE](LICENSE).
