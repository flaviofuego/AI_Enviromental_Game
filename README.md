# 🏒 Hockey is Melting Down - RL Air Hockey Game

![Hockey Game](pages/hockey_banner.jpg)

## 📋 Descripción

**Hockey is Melting Down** es un juego de hockey de aire (Air Hockey) que utiliza Reinforcement Learning para entrenar agentes que juegan de manera inteligente contra humanos. El proyecto aborda problemas específicos del entrenamiento de IA, como el movimiento vertical limitado, y proporciona soluciones mejoradas de aprendizaje.

## ✨ Características principales

### 🎮 Juego

- Interfaz gráfica completa con Pygame
- Controles intuitivos para juego humano vs IA
- Sistema de puntuación y niveles de dificultad
- Física realista de puck y mallets

### 🤖 Sistema de IA mejorado

- **Sistema de recompensas avanzado**: Incentivos balanceados para comportamientos deseados
- **Oponente inteligente**: 6 niveles progresivos de dificultad
- **Observaciones expandidas**: 21 dimensiones de datos para toma de decisiones
- **Curriculum Learning**: Adaptación automática de dificultad según rendimiento
- **Hiperparámetros optimizados**: Configuraciones de red neuronal mejoradas

## 🚀 Instalación

### Requisitos

- Python 3.12 o superior
- Dependencias en pyproject.toml

### Instalación rápida

```bash
# Clonar el repositorio
git clone https://github.com/yourusername/AI_Enviromental_Game.git
cd AI_Enviromental_Game

# Configurar el entorno (creará directorios necesarios y scripts)
python setup.py
```

### Dependencias principales

- gymnasium
- matplotlib
- numpy
- pandas
- pygame
- seaborn
- stable-baselines3 (con extras)
- tensorboard
- torch

## 📊 Uso

### Análisis de modelos

```bash
python quick_analysis.py
```

### Jugar con modelo entrenado

```bash
python main.py
```

### Entrenamiento personalizado

```bash
python quick_train_model.py --model_path models/your_model.zip
# Seguir las opciones del menú
```

## 🏆 Resultados esperados

| Métrica | Modelo Original | Modelo Mejorado |
|---------|----------------|-----------------|
| **Movimiento Vertical** | 0.0% ❌ | **94.8%** ✅ |
| **Movimiento Horizontal** | 80.1% | 5.1% |
| **Stay (Quieto)** | 19.9% | 0.0% ✅ |

## 🔧 Estructura del proyecto

```plaintext
.
├── air_hockey_env.py           # Entorno básico
├── compare_models/             # Scripts para comparar modelos
├── constants.py                # Constantes del juego
├── game/                       # Componentes adicionales del juego
├── improved_logs/              # Logs de entrenamiento mejorado
├── improved_models/            # Modelos entrenados mejorados
├── improved_training_system.py # Sistema de entrenamiento mejorado
├── info/                       # Documentación y soluciones
├── main.py                     # Punto de entrada principal
├── models/                     # Modelos entrenados originales
├── pages/                      # Recursos de UI
├── pyproject.toml              # Configuración del proyecto
├── quick_train_model.py        # Entrenamiento rápido
├── README.md                   # Este archivo
├── setup.py                    # Script de configuración
├── sprites.py                  # Sprites del juego
├── table.py                    # Tabla de hockey
└── utils.py                    # Funciones de utilidad
```

## 🔍 Debugging y solución de problemas

### Problemas comunes

- **El agente no mejora**:
  - Ajustar learning rate entre 1e-4 y 5e-4
  - Verificar que el entrenamiento empiece en dificultad 0
  - Revisar balance de recompensas en logs

- **Entrenamiento muy lento**:
  - Reducir
