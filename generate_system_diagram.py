import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow
import numpy as np

# Configurar el estilo y tamaño de la figura
plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Colores del tema
color_env = '#3498db'      # Azul
color_agent = '#2ecc71'    # Verde
color_training = '#e74c3c' # Rojo
color_game = '#f39c12'     # Naranja
color_arrow = '#34495e'    # Gris oscuro
color_data = '#9b59b6'     # Púrpura

# Título
ax.text(7, 9.5, 'Arquitectura del Sistema Air Hockey RL', 
        fontsize=20, fontweight='bold', ha='center')

# 1. Componente del Entorno (Gymnasium)
env_box = FancyBboxPatch((0.5, 6), 3, 2.5, 
                          boxstyle="round,pad=0.1",
                          facecolor=color_env, 
                          edgecolor='black',
                          alpha=0.8,
                          linewidth=2)
ax.add_patch(env_box)
ax.text(2, 7.7, 'Entorno Gymnasium', fontsize=12, fontweight='bold', ha='center', color='white')
ax.text(2, 7.3, 'AirHockeyEnv', fontsize=10, ha='center', color='white')
ax.text(2, 6.9, '• Espacio de observación: 13-21D', fontsize=8, ha='center', color='white')
ax.text(2, 6.6, '• Espacio de acción: 5 discretas', fontsize=8, ha='center', color='white')
ax.text(2, 6.3, '• Sistema de recompensas', fontsize=8, ha='center', color='white')

# 2. Componentes del Juego
game_box = FancyBboxPatch((5, 6), 3.5, 2.5,
                          boxstyle="round,pad=0.1",
                          facecolor=color_game,
                          edgecolor='black',
                          alpha=0.8,
                          linewidth=2)
ax.add_patch(game_box)
ax.text(6.75, 7.7, 'Motor del Juego', fontsize=12, fontweight='bold', ha='center', color='white')
ax.text(6.75, 7.3, 'Pygame Engine', fontsize=10, ha='center', color='white')
ax.text(6.75, 6.9, '• Física: colisiones, fricción', fontsize=8, ha='center', color='white')
ax.text(6.75, 6.6, '• Renderizado 60 FPS', fontsize=8, ha='center', color='white')
ax.text(6.75, 6.3, '• Sprites: Puck, Mallets', fontsize=8, ha='center', color='white')

# 3. Agente PPO
agent_box = FancyBboxPatch((10, 6), 3, 2.5,
                           boxstyle="round,pad=0.1",
                           facecolor=color_agent,
                           edgecolor='black',
                           alpha=0.8,
                           linewidth=2)
ax.add_patch(agent_box)
ax.text(11.5, 7.7, 'Agente PPO', fontsize=12, fontweight='bold', ha='center', color='white')
ax.text(11.5, 7.3, 'Stable Baselines3', fontsize=10, ha='center', color='white')
ax.text(11.5, 6.9, '• Red: [128, 128]', fontsize=8, ha='center', color='white')
ax.text(11.5, 6.6, '• Learning rate: 3e-4', fontsize=8, ha='center', color='white')
ax.text(11.5, 6.3, '• Batch size: 64', fontsize=8, ha='center', color='white')

# 4. Sistema de Entrenamiento
train_box = FancyBboxPatch((5, 3), 4, 2,
                           boxstyle="round,pad=0.1",
                           facecolor=color_training,
                           edgecolor='black',
                           alpha=0.8,
                           linewidth=2)
ax.add_patch(train_box)
ax.text(7, 4.5, 'Sistema de Entrenamiento', fontsize=12, fontweight='bold', ha='center', color='white')
ax.text(7, 4.0, '• Curriculum Learning', fontsize=9, ha='center', color='white')
ax.text(7, 3.6, '• Callbacks de monitoreo', fontsize=9, ha='center', color='white')
ax.text(7, 3.2, '• Ajuste dinámico de dificultad', fontsize=9, ha='center', color='white')

# 5. Oponente Adaptativo
opponent_box = FancyBboxPatch((0.5, 3), 3, 2,
                              boxstyle="round,pad=0.1",
                              facecolor='#8e44ad',
                              edgecolor='black',
                              alpha=0.8,
                              linewidth=2)
ax.add_patch(opponent_box)
ax.text(2, 4.5, 'Oponente Inteligente', fontsize=12, fontweight='bold', ha='center', color='white')
ax.text(2, 4.0, '• 6 niveles de dificultad', fontsize=9, ha='center', color='white')
ax.text(2, 3.6, '• Comportamiento adaptativo', fontsize=9, ha='center', color='white')
ax.text(2, 3.2, '• Estrategias O/D', fontsize=9, ha='center', color='white')

# 6. Sistema de Análisis
analysis_box = FancyBboxPatch((10, 3), 3, 2,
                              boxstyle="round,pad=0.1",
                              facecolor='#16a085',
                              edgecolor='black',
                              alpha=0.8,
                              linewidth=2)
ax.add_patch(analysis_box)
ax.text(11.5, 4.5, 'Análisis y Métricas', fontsize=12, fontweight='bold', ha='center', color='white')
ax.text(11.5, 4.0, '• Tensorboard logs', fontsize=9, ha='center', color='white')
ax.text(11.5, 3.6, '• Evaluación periódica', fontsize=9, ha='center', color='white')
ax.text(11.5, 3.2, '• Visualización', fontsize=9, ha='center', color='white')

# Flujo de datos - Flechas
# Env -> Game
ax.annotate('', xy=(5, 7.25), xytext=(3.5, 7.25),
            arrowprops=dict(arrowstyle='->', lw=2, color=color_arrow))
ax.text(4.25, 7.4, 'Estados', fontsize=8, ha='center')

# Game -> Agent
ax.annotate('', xy=(10, 7.25), xytext=(8.5, 7.25),
            arrowprops=dict(arrowstyle='->', lw=2, color=color_arrow))
ax.text(9.25, 7.4, 'Observaciones', fontsize=8, ha='center')

# Agent -> Env
ax.annotate('', xy=(2, 6), xytext=(11.5, 6),
            arrowprops=dict(arrowstyle='->', lw=2, color=color_arrow, 
                          connectionstyle="arc3,rad=-.4"))
ax.text(6.75, 5.3, 'Acciones', fontsize=8, ha='center')

# Training bidireccional con todos
ax.annotate('', xy=(7, 5), xytext=(2, 5),
            arrowprops=dict(arrowstyle='<->', lw=2, color=color_arrow))
ax.annotate('', xy=(9, 4), xytext=(11.5, 5),
            arrowprops=dict(arrowstyle='<->', lw=2, color=color_arrow,
                          connectionstyle="arc3,rad=.3"))
ax.annotate('', xy=(5, 4), xytext=(2, 5),
            arrowprops=dict(arrowstyle='<->', lw=2, color=color_arrow,
                          connectionstyle="arc3,rad=-.3"))

# Componentes adicionales
# Base de datos de experiencias
exp_circle = Circle((7, 0.8), 0.8, facecolor=color_data, edgecolor='black', linewidth=2)
ax.add_patch(exp_circle)
ax.text(7, 0.8, 'Buffer\nExp.', fontsize=9, ha='center', va='center', color='white', fontweight='bold')

# Conexión con buffer
ax.annotate('', xy=(7, 1.6), xytext=(7, 3),
            arrowprops=dict(arrowstyle='<->', lw=2, color=color_arrow))

# Leyenda de componentes
legend_y = 1.5
ax.text(0.5, legend_y, 'Componentes:', fontsize=10, fontweight='bold')
components = [
    ('Núcleo del juego', color_game),
    ('Entorno RL', color_env),
    ('Agente inteligente', color_agent),
    ('Sistema auxiliar', color_training)
]

for i, (label, color) in enumerate(components):
    rect = Rectangle((0.5 + i*3.2, legend_y - 0.4), 0.3, 0.2, 
                     facecolor=color, edgecolor='black')
    ax.add_patch(rect)
    ax.text(0.9 + i*3.2, legend_y - 0.3, label, fontsize=8, va='center')

# Información adicional
info_text = """
Flujo de datos:
1. El entorno genera observaciones del estado del juego
2. El agente PPO procesa las observaciones y selecciona acciones
3. Las acciones se ejecutan en el motor del juego (Pygame)
4. Se calculan recompensas basadas en el resultado
5. El sistema de entrenamiento ajusta la política del agente
6. El oponente adaptativo incrementa su dificultad según el progreso
"""

ax.text(0.5, -0.5, info_text, fontsize=8, va='top', 
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.5))

plt.tight_layout()
plt.savefig('diagrama_sistema.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('diagrama_sistema.pdf', bbox_inches='tight', facecolor='white')
print("Diagrama del sistema generado exitosamente: diagrama_sistema.png y diagrama_sistema.pdf")
plt.show() 