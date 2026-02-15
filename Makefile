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
# Entrenamiento IA  (v2: 9 acciones con diagonales)
# ---------------------------------------------------------------------------
# Presets disponibles:
#   v2_quick     — ~100k steps, pruebas rápidas
#   v2_standard  — ~500k steps, entrenamiento balanceado (default)
#   v2_deep      — ~2M  steps, entrenamiento exhaustivo
#   quick / standard / deep / exploration — presets legacy (4 acciones)
#
# Entornos:  base | powerups
#
# Overrides opcionales (se aplican sobre cualquier preset):
#   --timesteps N    --epochs N       --lr F
#   --batch-size N   --gamma F        --ent-coef F
#   --clip-range F   --n-steps N      --n-envs N
#   --checkpoint-freq N  --eval-freq N
# ---------------------------------------------------------------------------

.PHONY: train
train: ## Entrenar modelo IA (v2_standard, ~500k steps)
	uv run python -m training.train --preset v2_standard --env base

.PHONY: train-quick
train-quick: ## Entrenamiento rápido para pruebas (~100k steps)
	uv run python -m training.train --preset v2_quick --env base

.PHONY: train-deep
train-deep: ## Entrenamiento profundo (~2M steps)
	uv run python -m training.train --preset v2_deep --env base

.PHONY: train-powerups
train-powerups: ## Entrenar con power-ups (obs 26-dim)
	uv run python -m training.train --preset v2_standard --env powerups

.PHONY: train-parallel
train-parallel: ## Entrenar con 4 entornos paralelos
	uv run python -m training.train --preset v2_standard --env base --n-envs 4

.PHONY: train-custom
train-custom: ## Entrenar con parámetros personalizados (ejemplo)
	uv run python -m training.train --preset v2_standard --env base \
		--timesteps 300000 --lr 3e-4 --ent-coef 0.02

.PHONY: train-resume
train-resume: ## Reanudar entrenamiento desde último checkpoint
	uv run python -m training.train --preset v2_standard --env base --resume auto

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
