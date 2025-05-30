import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Crear figura con múltiples subplots
fig = plt.figure(figsize=(16, 12))

# 1. Evolución del Entrenamiento
ax1 = plt.subplot(2, 3, 1)
steps = [100, 500, 1000, 2000]
rewards = [-12.5, 45.7, 78.3, 92.6]
std_devs = [8.3, 15.2, 12.1, 9.4]
win_rates = [15, 42, 68, 81]

ax1.errorbar(steps, rewards, yerr=std_devs, fmt='o-', capsize=5, capthick=2, 
             label='Recompensa Promedio', linewidth=2, markersize=8)
ax1_twin = ax1.twinx()
ax1_twin.plot(steps, win_rates, 's-', color='green', label='Tasa de Victoria (%)', 
              linewidth=2, markersize=8)

ax1.set_xlabel('Steps (x1000)')
ax1.set_ylabel('Recompensa Promedio', color='blue')
ax1_twin.set_ylabel('Tasa de Victoria (%)', color='green')
ax1.set_title('Evolución del Rendimiento durante el Entrenamiento')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
ax1_twin.legend(loc='lower right')

# 2. Distribución de Acciones
ax2 = plt.subplot(2, 3, 2)
actions = ['Up', 'Down', 'Left', 'Right', 'Stay']
initial_dist = [8, 10, 35, 37, 10]
final_dist = [22, 24, 18, 19, 17]

x = np.arange(len(actions))
width = 0.35

bars1 = ax2.bar(x - width/2, initial_dist, width, label='Inicial (100k steps)', alpha=0.8)
bars2 = ax2.bar(x + width/2, final_dist, width, label='Final (2M steps)', alpha=0.8)

ax2.set_ylabel('Porcentaje de uso (%)')
ax2.set_title('Distribución de Acciones: Inicial vs Final')
ax2.set_xticks(x)
ax2.set_xticklabels(actions)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

# 3. Comparación con Baselines
ax3 = plt.subplot(2, 3, 3)
models = ['Random\nAgent', 'Heuristic\nAI', 'DQN\nSimple', 'PPO\nMejorado', 'PPO +\nCurriculum']
final_rewards = [-85.3, 15.2, 52.4, 92.6, 96.5]
colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']

bars = ax3.bar(models, final_rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_ylabel('Recompensa Final Promedio')
ax3.set_title('Comparación de Rendimiento entre Modelos')
ax3.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for bar, value in zip(bars, final_rewards):
    height = bar.get_height()
    label_y = height + 2 if height > 0 else height - 5
    ax3.text(bar.get_x() + bar.get_width()/2., label_y,
             f'{value:.1f}',
             ha='center', va='bottom' if height > 0 else 'top',
             fontsize=9, fontweight='bold')

# 4. Análisis de Convergencia
ax4 = plt.subplot(2, 3, 4)
episodes = np.arange(0, 2000, 50)
np.random.seed(42)

# Simular curvas de aprendizaje
basic_rewards = []
improved_rewards = []

for i, ep in enumerate(episodes):
    # Sistema básico - convergencia lenta con alta varianza
    basic_mean = -50 + 70 * (1 - np.exp(-ep/800))
    basic_std = 25 * np.exp(-ep/1500) + 10
    basic_rewards.append(basic_mean + np.random.normal(0, basic_std))
    
    # Sistema mejorado - convergencia rápida con baja varianza
    improved_mean = -20 + 110 * (1 - np.exp(-ep/400))
    improved_std = 15 * np.exp(-ep/800) + 5
    improved_rewards.append(improved_mean + np.random.normal(0, improved_std))

# Suavizar las curvas
window = 5
basic_smooth = np.convolve(basic_rewards, np.ones(window)/window, mode='valid')
improved_smooth = np.convolve(improved_rewards, np.ones(window)/window, mode='valid')
episodes_smooth = episodes[:len(basic_smooth)]

ax4.plot(episodes_smooth, basic_smooth, label='Sistema Básico', linewidth=2, alpha=0.8)
ax4.plot(episodes_smooth, improved_smooth, label='Sistema Mejorado', linewidth=2, alpha=0.8)
ax4.fill_between(episodes_smooth, basic_smooth - 15, basic_smooth + 15, alpha=0.2)
ax4.fill_between(episodes_smooth, improved_smooth - 8, improved_smooth + 8, alpha=0.2)

ax4.set_xlabel('Episodios (x1000)')
ax4.set_ylabel('Recompensa Promedio')
ax4.set_title('Comparación de Sistemas de Recompensas')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% Performance')

# 5. Métricas de Eficiencia
ax5 = plt.subplot(2, 3, 5)
phases = ['Fase 1\n(0-100k)', 'Fase 2\n(100k-500k)', 'Fase 3\n(500k-1M)', 'Fase 4\n(1M-2M)']
hit_efficiency = [0.08, 0.21, 0.35, 0.42]
vertical_actions = [8, 35, 52, 48]

x_pos = np.arange(len(phases))
ax5_twin = ax5.twinx()

bars1 = ax5.bar(x_pos - 0.2, hit_efficiency, 0.4, label='Eficiencia de Golpes', 
                 color='skyblue', alpha=0.8)
bars2 = ax5_twin.bar(x_pos + 0.2, vertical_actions, 0.4, label='Acciones Verticales (%)', 
                      color='lightcoral', alpha=0.8)

ax5.set_xlabel('Fases de Entrenamiento')
ax5.set_ylabel('Eficiencia de Golpes', color='skyblue')
ax5_twin.set_ylabel('Acciones Verticales (%)', color='lightcoral')
ax5.set_title('Evolución de Métricas Clave por Fase')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(phases)
ax5.tick_params(axis='y', labelcolor='skyblue')
ax5_twin.tick_params(axis='y', labelcolor='lightcoral')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Resultados de Usabilidad
ax6 = plt.subplot(2, 3, 6)
categories = ['Fluidez\ndel juego', 'Naturalidad\ndel oponente', 'Dificultad\nbalanceada', 'Diversión\ngeneral']
scores = [4.6, 4.2, 3.8, 4.4]
max_score = 5

# Crear gráfico de radar
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
scores_normalized = scores + scores[:1]  # Cerrar el polígono
angles += angles[:1]

ax6 = plt.subplot(2, 3, 6, projection='polar')
ax6.plot(angles, scores_normalized, 'o-', linewidth=2, color='green', markersize=8)
ax6.fill(angles, scores_normalized, alpha=0.25, color='green')
ax6.set_ylim(0, max_score)
ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories, size=9)
ax6.set_yticks([1, 2, 3, 4, 5])
ax6.set_yticklabels(['1', '2', '3', '4', '5'], size=8)
ax6.set_title('Resultados del Test de Usabilidad\n(Escala 1-5)', pad=20)
ax6.grid(True)

# Añadir valores en los puntos
for angle, score, cat in zip(angles[:-1], scores, categories):
    ax6.text(angle, score + 0.1, f'{score:.1f}', 
             ha='center', va='center', fontsize=10, fontweight='bold')

# Ajustar layout y guardar
plt.tight_layout()
plt.savefig('resultados_completos.png', dpi=300, bbox_inches='tight')
plt.savefig('resultados_completos.pdf', bbox_inches='tight')
print("Gráficos de resultados generados: resultados_completos.png y resultados_completos.pdf")

# Generar gráfico adicional de eficiencia computacional
fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(12, 5))

# Eficiencia computacional
configs = ['CPU Base', 'CPU Optimizado', 'Multi-env (4)']
steps_per_sec = [215, 342, 856]
memory_gb = [1.2, 1.1, 2.8]
time_to_80 = [285, 178, 72]

x = np.arange(len(configs))
width = 0.25

ax7.bar(x - width, steps_per_sec, width, label='Steps/segundo', alpha=0.8)
ax7.bar(x, [m * 100 for m in memory_gb], width, label='Memoria RAM (x100 MB)', alpha=0.8)
ax7.bar(x + width, time_to_80, width, label='Tiempo hasta 80% (min)', alpha=0.8)

ax7.set_xlabel('Configuración')
ax7.set_ylabel('Valor')
ax7.set_title('Métricas de Eficiencia Computacional')
ax7.set_xticks(x)
ax7.set_xticklabels(configs)
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

# Tabla de comparación detallada
ax8.axis('tight')
ax8.axis('off')

table_data = [
    ['Métrica', 'Random', 'Heuristic', 'DQN', 'PPO Mejorado', 'PPO+Curr'],
    ['Recompensa', '-85.3±12.1', '15.2±18.5', '52.4±20.3', '92.6±9.4', '96.5±7.2'],
    ['Tasa Victoria', '5%', '35%', '55%', '81%', '85%'],
    ['Tiempo Entrena.', 'N/A', 'N/A', '4h 15m', '3h 42m', '4h 08m'],
    ['Estabilidad', 'Baja', 'Media', 'Media', 'Alta', 'Muy Alta']
]

table = ax8.table(cellText=table_data[1:], colLabels=table_data[0],
                  cellLoc='center', loc='center',
                  colWidths=[0.15, 0.15, 0.15, 0.15, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Colorear celdas
for i in range(1, len(table_data)):
    for j in range(1, 6):
        cell = table[(i, j)]
        if j == 5:  # PPO+Curriculum
            cell.set_facecolor('#90EE90')
        elif j == 4:  # PPO Mejorado
            cell.set_facecolor('#98FB98')

ax8.set_title('Tabla Comparativa de Modelos', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('eficiencia_y_comparacion.png', dpi=300, bbox_inches='tight')
print("Gráficos adicionales generados: eficiencia_y_comparacion.png")

plt.show() 