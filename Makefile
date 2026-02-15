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
# Entrenamiento IA
# ---------------------------------------------------------------------------

.PHONY: train
train: ## Entrenar modelo IA (preset standard)
	uv run python -m training.train --preset standard --env base

.PHONY: train-quick
train-quick: ## Entrenamiento rápido para pruebas
	uv run python -m training.train --preset quick --env base

.PHONY: train-deep
train-deep: ## Entrenamiento profundo (largo)
	uv run python -m training.train --preset deep --env base

.PHONY: train-resume
train-resume: ## Reanudar entrenamiento desde último checkpoint
	uv run python -m training.train --preset standard --env base --resume auto

# ---------------------------------------------------------------------------
# Evaluación y análisis
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Ejecutar tests de integración
	uv run python -m pytest tests/ -v

.PHONY: check-models
check-models: ## Verificar modelos disponibles y su estado
	uv run python -m tests.check_models

.PHONY: tensorboard
tensorboard: ## Lanzar TensorBoard para ver métricas de entrenamiento
	uv run tensorboard --logdir logs/ --port 6006

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

.PHONY: lint
lint: ## Verificar estilo de código
	uv run python -m py_compile game/main.py
	uv run python -m py_compile game/core/game_engine.py
	uv run python -m py_compile training/envs/base_env.py
	@echo "✓ Archivos principales compilan correctamente"

.PHONY: clean
clean: ## Limpiar archivos temporales y cachés
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/

.PHONY: clean-models
clean-models: ## Eliminar modelos entrenados (¡CUIDADO!)
	@echo "⚠️  Esto eliminará TODOS los modelos entrenados."
	@read -p "¿Estás seguro? (y/N): " confirm && [ "$$confirm" = "y" ] && \
		rm -rf models/ improved_models/ logs/ improved_logs/ || \
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
