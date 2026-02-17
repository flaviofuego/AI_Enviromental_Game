# ============================================================================
# Makefile — Hockey Is Melting Down
# Usa `uv` como gestor de paquetes y entorno virtual.
# ============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ---------------------------------------------------------------------------
# Instalación y entorno
# ---------------------------------------------------------------------------

.PHONY: install
install: ## Instalar todas las dependencias con uv
	uv sync

.PHONY: install-dev
install-dev: ## Instalar dependencias + herramientas de desarrollo
	uv sync --all-extras

# ---------------------------------------------------------------------------
# Juego
# ---------------------------------------------------------------------------

.PHONY: play
play: ## Lanzar el juego (menú principal + selección de niveles)
	uv run python -m game.main_hub

.PHONY: play-quick
play-quick: ## Iniciar partida rápida (nivel 1, Player vs IA)
	uv run python -m game.main

.PHONY: play-level
play-level: ## Jugar un nivel específico: make play-level LEVEL=3
	uv run python -c "import pygame; pygame.init(); \
		s = pygame.display.set_mode((1200,800)); \
		from game.main import main_with_config; \
		main_with_config(screen=s, level_id=$(LEVEL)); \
		pygame.quit()"

# ---------------------------------------------------------------------------
# Entrenamiento IA  (Typer CLI con Rich)
# ---------------------------------------------------------------------------
# Uso principal:
#   make train           — Wizard interactivo (selección visual de todo)
#   make train-start     — Entrenar con preset por defecto (v3_optimized)
#   make train-presets   — Ver todos los presets disponibles
#
# Atajos de presets:
#   make train-quick     — ~800K steps, pruebas rápidas
#   make train-deep      — ~5M steps, exhaustivo
#   make train-dqn       — DQN 2M steps
#
# Entrenamiento directo con parámetros:
#   make train-custom PRESET=v3_optimized ENVS=4 TIMESTEPS=1000000
#
# Entornos:  base | powerups
#
# Variables disponibles para train-custom:
#   PRESET     (default: v3_optimized)
#   ENV        (default: base)
#   ENVS       (default: 4)
#   TIMESTEPS  (opcional, override)
#   LR         (opcional, override)
#   BATCH      (opcional, override)
# ---------------------------------------------------------------------------

# Variables con defaults
PRESET  ?= v3_optimized
ENV     ?= base
ENVS    ?= 4

# Construir overrides dinámicamente
_OVERRIDES :=
ifdef TIMESTEPS
_OVERRIDES += --timesteps $(TIMESTEPS)
endif
ifdef LR
_OVERRIDES += --lr $(LR)
endif
ifdef BATCH
_OVERRIDES += --batch-size $(BATCH)
endif
ifdef EPOCHS
_OVERRIDES += --epochs $(EPOCHS)
endif
ifdef GAMMA
_OVERRIDES += --gamma $(GAMMA)
endif
ifdef ENT_COEF
_OVERRIDES += --ent-coef $(ENT_COEF)
endif

.PHONY: train
train: ## 🧙 Wizard interactivo — selección visual de todos los parámetros
	uv run python -m training.train wizard

.PHONY: train-start
train-start: ## Entrenar con preset (v3_optimized por defecto)
	uv run python -m training.train start --preset $(PRESET) --env $(ENV) --n-envs $(ENVS) $(_OVERRIDES)

.PHONY: train-quick
train-quick: ## Entrenamiento rápido (~800K steps, ~5 min)
	uv run python -m training.train start --preset v3_quick --env base --n-envs $(ENVS)

.PHONY: train-deep
train-deep: ## Entrenamiento profundo (~5M steps, ~1h)
	uv run python -m training.train start --preset v3_deep --env base --n-envs $(ENVS)

.PHONY: train-dqn
train-dqn: ## Entrenar con DQN (~2M steps)
	uv run python -m training.train start --preset v3_dqn --env base --n-envs $(ENVS)

.PHONY: train-powerups
train-powerups: ## Entrenar con power-ups
	uv run python -m training.train start --preset $(PRESET) --env powerups --n-envs $(ENVS)

.PHONY: train-custom
train-custom: ## Entrenar con parámetros personalizados (ver variables arriba)
	uv run python -m training.train start --preset $(PRESET) --env $(ENV) --n-envs $(ENVS) $(_OVERRIDES)

.PHONY: train-resume
train-resume: ## Reanudar entrenamiento desde último checkpoint
	uv run python -m training.train start --preset $(PRESET) --env $(ENV) --resume auto

.PHONY: train-presets
train-presets: ## Mostrar tabla de presets disponibles
	uv run python -m training.train presets

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Ejecutar todos los tests
	uv run pytest tests/ -v

.PHONY: test-fast
test-fast: ## Tests rápidos (excluye SB3 check_env)
	uv run pytest tests/ -v -k "not SB3"

.PHONY: test-integration
test-integration: ## Solo tests de integración
	uv run pytest tests/test_integration.py -v

# ---------------------------------------------------------------------------
# Evaluación y análisis
# ---------------------------------------------------------------------------

.PHONY: tensorboard
tensorboard: ## Lanzar TensorBoard para ver métricas de entrenamiento
	uv run tensorboard --logdir logs/ --port 6006

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

.PHONY: lint
lint: ## Verificar estilo de código
	@uv run python -m py_compile game/main.py
	@uv run python -m py_compile game/core/game_engine.py
	@uv run python -m py_compile training/envs/base_env.py
	@uv run python -m py_compile training/train.py
	@uv run python -m py_compile training/__main__.py
	@uv run python -m py_compile game/ai/model_loader.py
	@echo "✓ Archivos principales compilan correctamente"

.PHONY: clean
clean: ## Limpiar archivos temporales y cachés
	find . -not -path './.venv/*' -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -not -path './.venv/*' -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/

.PHONY: clean-models
clean-models: ## Eliminar modelos entrenados (¡CUIDADO!)
	@echo "⚠️  Esto eliminará TODOS los modelos entrenados."
	@read -p "¿Estás seguro? (y/N): " confirm && [ "$$confirm" = "y" ] && \
		rm -rf models/ logs/ || \
		echo "Cancelado."

# ---------------------------------------------------------------------------
# Ayuda
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Mostrar esta ayuda
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║       Hockey Is Melting Down — Comandos Disponibles     ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
