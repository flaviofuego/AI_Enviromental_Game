# analyze_fixed_model.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import json
import os
from datetime import datetime
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from improved_training_system_fixed import FixedAirHockeyEnv

# Configurar matplotlib para mejor visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FixedModelAnalyzer:
    """Analizador completo para el modelo fixed de Air Hockey"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = PPO.load(model_path)
        self.env = FixedAirHockeyEnv()
        
        # Métricas a recolectar
        self.metrics = {
            # Métricas de acciones
            'actions': defaultdict(int),
            'action_sequences': defaultdict(int),
            'action_transitions': defaultdict(lambda: defaultdict(int)),
            
            # Métricas de posición
            'positions': {'x': [], 'y': []},
            'position_heatmap': np.zeros((20, 20)),
            'position_zones': defaultdict(int),
            
            # Métricas de rendimiento
            'episodes': [],
            'rewards': [],
            'episode_lengths': [],
            'goals_scored': [],
            'goals_conceded': [],
            'hits': [],
            'win_rate': 0,
            
            # Métricas de comportamiento
            'distance_to_puck': [],
            'time_in_offensive_half': 0,
            'time_in_defensive_half': 0,
            'defensive_saves': 0,
            'offensive_attempts': 0,
            
            # Métricas de velocidad y movimiento
            'velocities': [],
            'acceleration_patterns': [],
            'movement_efficiency': [],
            
            # Métricas temporales
            'time_to_first_hit': [],
            'time_between_hits': [],
            'reaction_times': [],
            
            # Métricas estratégicas
            'puck_control_time': 0,
            'positioning_quality': [],
            'defensive_positioning': [],
            'offensive_positioning': []
        }
        
        self.episode_data = []
        self.current_episode = None
        
    def analyze(self, num_episodes=100, verbose=True):
        """Realizar análisis completo del modelo"""
        print(f"\n{'='*70}")
        print(f"ANÁLISIS COMPLETO DEL MODELO: {self.model_path}")
        print(f"{'='*70}\n")
        
        start_time = datetime.now()
        
        # Ejecutar episodios de prueba
        for episode in range(num_episodes):
            if verbose and episode % 10 == 0:
                print(f"Analizando episodio {episode+1}/{num_episodes}...")
            
            self._run_episode()
        
        # Calcular estadísticas finales
        self._calculate_final_statistics()
        
        # Generar reporte
        self._generate_report()
        
        # Crear visualizaciones
        self._create_visualizations()
        
        end_time = datetime.now()
        print(f"\n⏱️  Análisis completado en: {end_time - start_time}")
        
        return self.metrics
    
    def _run_episode(self):
        """Ejecutar un episodio completo y recolectar métricas"""
        obs, _ = self.env.reset()
        done = False
        
        # Datos del episodio actual
        self.current_episode = {
            'rewards': [],
            'actions': [],
            'positions': [],
            'distances': [],
            'hits': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'steps': 0,
            'first_hit_step': None,
            'hit_times': []
        }
        
        last_hit_time = 0
        
        while not done:
            # Predecir acción
            action, _ = self.model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.ndim == 0 else int(action[0])
            
            # Ejecutar acción
            obs, reward, done, truncated, info = self.env.step(action)
            
            # Recolectar métricas
            self._collect_step_metrics(action, reward, info)
            
            # Detectar hits
            if info.get('hit_puck', False):
                self.current_episode['hits'] += 1
                if self.current_episode['first_hit_step'] is None:
                    self.current_episode['first_hit_step'] = self.current_episode['steps']
                
                if last_hit_time > 0:
                    self.metrics['time_between_hits'].append(
                        self.current_episode['steps'] - last_hit_time
                    )
                last_hit_time = self.current_episode['steps']
            
            self.current_episode['steps'] += 1
            
            if done or truncated:
                break
        
        # Procesar datos del episodio
        self._process_episode_data()
    
    def _collect_step_metrics(self, action, reward, info):
        """Recolectar métricas de cada paso"""
        # Acciones
        self.metrics['actions'][action] += 1
        self.current_episode['actions'].append(action)
        
        # Recompensas
        self.current_episode['rewards'].append(reward)
        
        # Posiciones
        ai_pos = self.env.ai_mallet_position.copy()
        self.current_episode['positions'].append(ai_pos)
        self.metrics['positions']['x'].append(ai_pos[0])
        self.metrics['positions']['y'].append(ai_pos[1])
        
        # Actualizar heatmap
        x_idx = int((ai_pos[0] / self.env.WIDTH) * 19)
        y_idx = int((ai_pos[1] / self.env.HEIGHT) * 19)
        x_idx = max(0, min(19, x_idx))
        y_idx = max(0, min(19, y_idx))
        self.metrics['position_heatmap'][y_idx, x_idx] += 1
        
        # Zonas
        if ai_pos[0] > self.env.WIDTH * 0.75:
            zone = 'defensive'
        elif ai_pos[0] > self.env.WIDTH * 0.5:
            zone = 'mid_defensive'
        else:
            zone = 'offensive'
        self.metrics['position_zones'][zone] += 1
        
        # Distancia al puck
        distance = self.env._calculate_distance_to_puck()
        self.current_episode['distances'].append(distance)
        self.metrics['distance_to_puck'].append(distance)
        
        # Tiempo en cada mitad
        if self.env.puck.position[0] > self.env.WIDTH / 2:
            self.metrics['time_in_defensive_half'] += 1
        else:
            self.metrics['time_in_offensive_half'] += 1
        
        # Velocidad
        if len(self.current_episode['positions']) > 1:
            prev_pos = self.current_episode['positions'][-2]
            velocity = np.sqrt(
                (ai_pos[0] - prev_pos[0])**2 + 
                (ai_pos[1] - prev_pos[1])**2
            )
            self.metrics['velocities'].append(velocity)
    
    def _process_episode_data(self):
        """Procesar datos del episodio completado"""
        episode = self.current_episode
        
        # Métricas básicas
        self.metrics['episodes'].append(len(self.metrics['episodes']) + 1)
        self.metrics['rewards'].append(sum(episode['rewards']))
        self.metrics['episode_lengths'].append(episode['steps'])
        
        # Goles
        goals_scored = self.env.ai_score
        goals_conceded = self.env.player_score
        self.metrics['goals_scored'].append(goals_scored)
        self.metrics['goals_conceded'].append(goals_conceded)
        
        # Hits
        self.metrics['hits'].append(episode['hits'])
        
        # Tiempo hasta primer hit
        if episode['first_hit_step'] is not None:
            self.metrics['time_to_first_hit'].append(episode['first_hit_step'])
        
        # Secuencias de acciones
        if len(episode['actions']) >= 3:
            for i in range(len(episode['actions']) - 2):
                seq = tuple(episode['actions'][i:i+3])
                self.metrics['action_sequences'][seq] += 1
        
        # Transiciones de acciones
        for i in range(len(episode['actions']) - 1):
            from_action = episode['actions'][i]
            to_action = episode['actions'][i+1]
            self.metrics['action_transitions'][from_action][to_action] += 1
        
        # Guardar datos completos del episodio
        self.episode_data.append({
            'episode': len(self.episode_data) + 1,
            'total_reward': sum(episode['rewards']),
            'steps': episode['steps'],
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'hits': episode['hits'],
            'avg_distance': np.mean(episode['distances']),
            'win': goals_scored > goals_conceded
        })
    
    def _calculate_final_statistics(self):
        """Calcular estadísticas finales"""
        # Win rate
        wins = sum(1 for ep in self.episode_data if ep['win'])
        self.metrics['win_rate'] = wins / len(self.episode_data) * 100
        
        # Eficiencia de movimiento
        if self.metrics['velocities']:
            avg_velocity = np.mean(self.metrics['velocities'])
            std_velocity = np.std(self.metrics['velocities'])
            self.metrics['movement_efficiency'] = {
                'avg_velocity': avg_velocity,
                'std_velocity': std_velocity,
                'max_velocity': max(self.metrics['velocities']),
                'stationary_percentage': sum(1 for v in self.metrics['velocities'] if v < 0.5) / len(self.metrics['velocities']) * 100
            }
        
        # Calidad de posicionamiento
        avg_distance = np.mean(self.metrics['distance_to_puck'])
        self.metrics['positioning_quality'] = {
            'avg_distance_to_puck': avg_distance,
            'defensive_time_percentage': self.metrics['time_in_defensive_half'] / (self.metrics['time_in_defensive_half'] + self.metrics['time_in_offensive_half']) * 100,
            'offensive_time_percentage': self.metrics['time_in_offensive_half'] / (self.metrics['time_in_defensive_half'] + self.metrics['time_in_offensive_half']) * 100
        }
    
    def _generate_report(self):
        """Generar reporte detallado"""
        print("\n" + "="*70)
        print("REPORTE DE ANÁLISIS DETALLADO")
        print("="*70)
        
        # 1. MÉTRICAS DE RENDIMIENTO
        print("\n📊 MÉTRICAS DE RENDIMIENTO")
        print("-"*50)
        print(f"Total de episodios analizados: {len(self.episode_data)}")
        print(f"Win Rate: {self.metrics['win_rate']:.1f}%")
        print(f"Recompensa promedio: {np.mean(self.metrics['rewards']):.2f} ± {np.std(self.metrics['rewards']):.2f}")
        print(f"Duración promedio de episodio: {np.mean(self.metrics['episode_lengths']):.1f} pasos")
        print(f"Goles anotados (promedio): {np.mean(self.metrics['goals_scored']):.2f}")
        print(f"Goles recibidos (promedio): {np.mean(self.metrics['goals_conceded']):.2f}")
        print(f"Hits por episodio: {np.mean(self.metrics['hits']):.2f}")
        
        # 2. DISTRIBUCIÓN DE ACCIONES
        print("\n🎮 DISTRIBUCIÓN DE ACCIONES")
        print("-"*50)
        action_names = ["Up", "Down", "Left", "Right", "Stay"]
        total_actions = sum(self.metrics['actions'].values())
        
        for i, name in enumerate(action_names):
            count = self.metrics['actions'][i]
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"{name:>6}: {count:>6} ({percentage:>5.1f}%) {bar}")
        
        # Análisis de movimiento
        vertical_actions = self.metrics['actions'][0] + self.metrics['actions'][1]
        horizontal_actions = self.metrics['actions'][2] + self.metrics['actions'][3]
        vertical_pct = (vertical_actions / total_actions) * 100
        horizontal_pct = (horizontal_actions / total_actions) * 100
        stay_pct = (self.metrics['actions'][4] / total_actions) * 100
        
        print(f"\nMovimiento Vertical: {vertical_pct:.1f}%")
        print(f"Movimiento Horizontal: {horizontal_pct:.1f}%")
        print(f"Sin movimiento: {stay_pct:.1f}%")
        
        # 3. ANÁLISIS POSICIONAL
        print("\n📍 ANÁLISIS POSICIONAL")
        print("-"*50)
        avg_x = np.mean(self.metrics['positions']['x'])
        avg_y = np.mean(self.metrics['positions']['y'])
        std_x = np.std(self.metrics['positions']['x'])
        std_y = np.std(self.metrics['positions']['y'])
        
        print(f"Posición promedio: ({avg_x:.1f}, {avg_y:.1f})")
        print(f"Desviación estándar: (σx={std_x:.1f}, σy={std_y:.1f})")
        
        print("\nDistribución por zonas:")
        total_zone_time = sum(self.metrics['position_zones'].values())
        for zone, count in sorted(self.metrics['position_zones'].items()):
            percentage = (count / total_zone_time) * 100
            print(f"  {zone}: {percentage:.1f}%")
        
        # 4. EFICIENCIA DE MOVIMIENTO
        print("\n🏃 EFICIENCIA DE MOVIMIENTO")
        print("-"*50)
        if 'movement_efficiency' in self.metrics and self.metrics['movement_efficiency']:
            eff = self.metrics['movement_efficiency']
            print(f"Velocidad promedio: {eff['avg_velocity']:.2f} ± {eff['std_velocity']:.2f}")
            print(f"Velocidad máxima: {eff['max_velocity']:.2f}")
            print(f"Tiempo estacionario: {eff['stationary_percentage']:.1f}%")
        
        # 5. MÉTRICAS TEMPORALES
        print("\n⏱️  MÉTRICAS TEMPORALES")
        print("-"*50)
        if self.metrics['time_to_first_hit']:
            avg_first_hit = np.mean(self.metrics['time_to_first_hit'])
            print(f"Tiempo promedio hasta primer hit: {avg_first_hit:.1f} pasos")
        
        if self.metrics['time_between_hits']:
            avg_between_hits = np.mean(self.metrics['time_between_hits'])
            print(f"Tiempo promedio entre hits: {avg_between_hits:.1f} pasos")
        
        # 6. CALIDAD DE POSICIONAMIENTO
        print("\n🎯 CALIDAD DE POSICIONAMIENTO")
        print("-"*50)
        if 'positioning_quality' in self.metrics:
            pos_q = self.metrics['positioning_quality']
            print(f"Distancia promedio al puck: {pos_q['avg_distance_to_puck']:.1f}")
            print(f"Tiempo en zona defensiva: {pos_q['defensive_time_percentage']:.1f}%")
            print(f"Tiempo en zona ofensiva: {pos_q['offensive_time_percentage']:.1f}%")
        
        # 7. SECUENCIAS DE ACCIONES MÁS COMUNES
        print("\n🔄 SECUENCIAS DE ACCIONES MÁS COMUNES")
        print("-"*50)
        top_sequences = sorted(self.metrics['action_sequences'].items(), 
                             key=lambda x: x[1], reverse=True)[:10]
        
        for seq, count in top_sequences:
            seq_names = [action_names[a] for a in seq]
            print(f"  {' → '.join(seq_names)}: {count} veces")
        
        # 8. EVALUACIÓN GENERAL
        print("\n⭐ EVALUACIÓN GENERAL")
        print("-"*50)
        
        # Evaluar rendimiento
        if self.metrics['win_rate'] > 70:
            performance = "EXCELENTE"
        elif self.metrics['win_rate'] > 50:
            performance = "BUENO"
        elif self.metrics['win_rate'] > 30:
            performance = "REGULAR"
        else:
            performance = "NECESITA MEJORA"
        
        print(f"Rendimiento general: {performance}")
        
        # Evaluar comportamiento
        if vertical_pct < 10:
            vertical_eval = "❌ CRÍTICO: Muy poco movimiento vertical"
        elif vertical_pct < 20:
            vertical_eval = "⚠️  ADVERTENCIA: Poco movimiento vertical"
        elif vertical_pct < 40:
            vertical_eval = "✅ BIEN: Buen balance de movimiento vertical"
        else:
            vertical_eval = "🏆 EXCELENTE: Movimiento muy equilibrado"
        
        print(f"Evaluación de movimiento: {vertical_eval}")
        
        # Evaluar posicionamiento
        if avg_y > self.env.HEIGHT * 0.7:
            pos_eval = "⚠️  Tiende a quedarse en la parte inferior"
        elif avg_y < self.env.HEIGHT * 0.3:
            pos_eval = "⚠️  Tiende a quedarse en la parte superior"
        else:
            pos_eval = "✅ Posicionamiento equilibrado"
        
        print(f"Evaluación de posición: {pos_eval}")
    
    def _create_visualizations(self):
        """Crear visualizaciones de las métricas"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"analysis_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Distribución de acciones
        plt.figure(figsize=(10, 6))
        action_names = ["Up", "Down", "Left", "Right", "Stay"]
        actions_data = [self.metrics['actions'][i] for i in range(5)]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7DC6F', '#BB8FCE']
        
        plt.subplot(2, 2, 1)
        plt.pie(actions_data, labels=action_names, colors=colors, autopct='%1.1f%%')
        plt.title('Distribución de Acciones')
        
        # 2. Evolución de recompensas
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['episodes'], self.metrics['rewards'], 'b-', alpha=0.6)
        plt.plot(self.metrics['episodes'], 
                pd.Series(self.metrics['rewards']).rolling(10).mean(), 
                'r-', linewidth=2, label='Media móvil (10 ep)')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa Total')
        plt.title('Evolución de Recompensas')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Heatmap de posiciones
        plt.subplot(2, 2, 3)
        sns.heatmap(self.metrics['position_heatmap'], 
                   cmap='YlOrRd', cbar_kws={'label': 'Frecuencia'})
        plt.title('Heatmap de Posiciones')
        plt.xlabel('X (normalizado)')
        plt.ylabel('Y (normalizado)')
        
        # 4. Goles anotados vs recibidos
        plt.subplot(2, 2, 4)
        episodes = range(len(self.metrics['goals_scored']))
        plt.plot(episodes, np.cumsum(self.metrics['goals_scored']), 
                'g-', label='Goles anotados', linewidth=2)
        plt.plot(episodes, np.cumsum(self.metrics['goals_conceded']), 
                'r-', label='Goles recibidos', linewidth=2)
        plt.xlabel('Episodio')
        plt.ylabel('Goles acumulados')
        plt.title('Progreso de Goles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Análisis de movimiento vertical
        plt.figure(figsize=(12, 8))
        
        # Histograma de posiciones Y
        plt.subplot(2, 2, 1)
        plt.hist(self.metrics['positions']['y'], bins=30, color='skyblue', edgecolor='black')
        plt.axvline(self.env.HEIGHT/2, color='red', linestyle='--', label='Centro')
        plt.xlabel('Posición Y')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Posiciones Verticales')
        plt.legend()
        
        # Gráfico de caja para posiciones
        plt.subplot(2, 2, 2)
        data_to_plot = [
            [y for y in self.metrics['positions']['y'] if y < self.env.HEIGHT * 0.33],
            [y for y in self.metrics['positions']['y'] if self.env.HEIGHT * 0.33 <= y < self.env.HEIGHT * 0.66],
            [y for y in self.metrics['positions']['y'] if y >= self.env.HEIGHT * 0.66]
        ]
        plt.boxplot(data_to_plot, labels=['Superior', 'Centro', 'Inferior'])
        plt.ylabel('Posición Y')
        plt.title('Distribución por Tercio del Campo')
        
        # Velocidades a lo largo del tiempo
        plt.subplot(2, 2, 3)
        if self.metrics['velocities']:
            plt.plot(self.metrics['velocities'][:1000], alpha=0.6)
            plt.axhline(np.mean(self.metrics['velocities']), 
                       color='red', linestyle='--', label='Promedio')
            plt.xlabel('Paso')
            plt.ylabel('Velocidad')
            plt.title('Velocidad de Movimiento')
            plt.legend()
        
        # Transiciones de acciones
        plt.subplot(2, 2, 4)
        transition_matrix = np.zeros((5, 5))
        for from_action in range(5):
            total = sum(self.metrics['action_transitions'][from_action].values())
            if total > 0:
                for to_action in range(5):
                    transition_matrix[from_action][to_action] = \
                        self.metrics['action_transitions'][from_action][to_action] / total
        
        sns.heatmap(transition_matrix, annot=True, fmt='.2f', 
                   xticklabels=action_names, yticklabels=action_names,
                   cmap='Blues')
        plt.title('Matriz de Transición de Acciones')
        plt.xlabel('Acción siguiente')
        plt.ylabel('Acción actual')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/movement_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Análisis temporal
        plt.figure(figsize=(10, 6))
        
        # Tiempo hasta primer hit
        if self.metrics['time_to_first_hit']:
            plt.subplot(2, 1, 1)
            plt.hist(self.metrics['time_to_first_hit'], bins=20, 
                    color='lightgreen', edgecolor='black')
            plt.axvline(np.mean(self.metrics['time_to_first_hit']), 
                       color='red', linestyle='--', label='Promedio')
            plt.xlabel('Pasos hasta primer hit')
            plt.ylabel('Frecuencia')
            plt.title('Distribución de Tiempo hasta Primer Hit')
            plt.legend()
        
        # Hits por episodio
        plt.subplot(2, 1, 2)
        plt.plot(self.metrics['episodes'], self.metrics['hits'], 'o-', alpha=0.6)
        plt.plot(self.metrics['episodes'], 
                pd.Series(self.metrics['hits']).rolling(10).mean(), 
                'r-', linewidth=2, label='Media móvil (10 ep)')
        plt.xlabel('Episodio')
        plt.ylabel('Número de hits')
        plt.title('Hits por Episodio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/temporal_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Guardar métricas en JSON
        metrics_to_save = {
            'model_path': self.model_path,
            'analysis_date': timestamp,
            'num_episodes': len(self.episode_data),
            'win_rate': self.metrics['win_rate'],
            'avg_reward': float(np.mean(self.metrics['rewards'])),
            'std_reward': float(np.std(self.metrics['rewards'])),
            'avg_goals_scored': float(np.mean(self.metrics['goals_scored'])),
            'avg_goals_conceded': float(np.mean(self.metrics['goals_conceded'])),
            'avg_hits': float(np.mean(self.metrics['hits'])),
            'action_distribution': {
                'up': self.metrics['actions'][0],
                'down': self.metrics['actions'][1],
                'left': self.metrics['actions'][2],
                'right': self.metrics['actions'][3],
                'stay': self.metrics['actions'][4]
            },
            'vertical_movement_percentage': float(
                (self.metrics['actions'][0] + self.metrics['actions'][1]) / 
                sum(self.metrics['actions'].values()) * 100
            ),
            'movement_efficiency': self.metrics.get('movement_efficiency', {}),
            'positioning_quality': self.metrics.get('positioning_quality', {})
        }
        
        with open(f"{output_dir}/analysis_metrics.json", 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        
        # Guardar datos de episodios en CSV
        df = pd.DataFrame(self.episode_data)
        df.to_csv(f"{output_dir}/episode_data.csv", index=False)
        
        print(f"\n📁 Resultados guardados en: {output_dir}/")
        print(f"   - metrics_overview.png: Resumen visual de métricas")
        print(f"   - movement_analysis.png: Análisis detallado de movimiento")
        print(f"   - temporal_analysis.png: Análisis temporal")
        print(f"   - analysis_metrics.json: Métricas en formato JSON")
        print(f"   - episode_data.csv: Datos de episodios en CSV")

def compare_models(model_paths, num_episodes=50):
    """Comparar múltiples modelos"""
    results = {}
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"\nAnalizando: {model_path}")
            analyzer = FixedModelAnalyzer(model_path)
            metrics = analyzer.analyze(num_episodes=num_episodes, verbose=False)
            
            model_name = os.path.basename(model_path).replace('.zip', '')
            results[model_name] = {
                'win_rate': metrics['win_rate'],
                'avg_reward': np.mean(metrics['rewards']),
                'vertical_movement': (metrics['actions'][0] + metrics['actions'][1]) / sum(metrics['actions'].values()) * 100,
                'avg_hits': np.mean(metrics['hits']),
                'avg_goals_scored': np.mean(metrics['goals_scored'])
            }
    
    # Crear tabla comparativa
    print("\n" + "="*80)
    print("COMPARACIÓN DE MODELOS")
    print("="*80)
    print(f"{'Modelo':<30} {'Win Rate':<10} {'Reward':<10} {'Vert.Mov%':<12} {'Hits/Ep':<10} {'Goals/Ep':<10}")
    print("-"*80)
    
    for model, metrics in results.items():
        print(f"{model:<30} {metrics['win_rate']:>8.1f}% {metrics['avg_reward']:>9.1f} "
              f"{metrics['vertical_movement']:>10.1f}% {metrics['avg_hits']:>9.1f} "
              f"{metrics['avg_goals_scored']:>9.1f}")
    
    return results

if __name__ == "__main__":
    print("🔬 ANALIZADOR DE MODELOS DE AIR HOCKEY")
    print("="*70)
    
    # Buscar modelos disponibles
    model_dirs = ["improved_models", "models"]
    available_models = []
    
    for dir in model_dirs:
        if os.path.exists(dir):
            for file in os.listdir(dir):
                if file.endswith('.zip'):
                    available_models.append(os.path.join(dir, file))
    
    if not available_models:
        print("❌ No se encontraron modelos para analizar")
        exit()
    
    print("\nModelos disponibles:")
    for i, model in enumerate(available_models):
        print(f"{i+1}. {model}")
    
    print(f"\n{len(available_models)+1}. Comparar todos los modelos")
    print(f"{len(available_models)+2}. Análisis rápido (10 episodios)")
    
    try:
        choice = int(input("\nSelecciona opción: "))
        
        if 1 <= choice <= len(available_models):
            # Analizar un modelo específico
            model_path = available_models[choice-1]
            num_episodes = int(input("Número de episodios a analizar (default 100): ") or "100")
            
            analyzer = FixedModelAnalyzer(model_path)
            analyzer.analyze(num_episodes=num_episodes)
            
        elif choice == len(available_models) + 1:
            # Comparar todos los modelos
            num_episodes = int(input("Episodios por modelo (default 50): ") or "50")
            compare_models(available_models, num_episodes=num_episodes)
            
        elif choice == len(available_models) + 2:
            # Análisis rápido
            model_choice = int(input(f"Selecciona modelo (1-{len(available_models)}): ")) - 1
            if 0 <= model_choice < len(available_models):
                analyzer = FixedModelAnalyzer(available_models[model_choice])
                analyzer.analyze(num_episodes=10)
            else:
                print("❌ Selección inválida")
        else:
            print("❌ Opción inválida")
            
    except ValueError:
        print("❌ Entrada inválida")
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}") 