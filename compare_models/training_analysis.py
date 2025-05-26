# training_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from ..improved_training_system import ImprovedAirHockeyEnv, create_improved_env
from air_hockey_env import AirHockeyEnv
import seaborn as sns

class ModelAnalyzer:
    """Clase para analizar y comparar el rendimiento de modelos de RL"""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model_comprehensive(self, model_path, model_name, env_type="improved", n_episodes=100):
        """Evaluación comprehensiva de un modelo"""
        print(f"\nEvaluando modelo: {model_name}")
        print(f"Archivo: {model_path}")
        
        try:
            # Cargar modelo
            model = PPO.load(model_path)
            
            # Crear entorno apropiado
            if env_type == "improved":
                env = ImprovedAirHockeyEnv()
            else:
                env = AirHockeyEnv()
            
            # Métricas a recopilar
            episode_rewards = []
            episode_lengths = []
            goals_scored = []
            goals_conceded = []
            hit_counts = []
            win_rates = []
            
            print(f"Ejecutando {n_episodes} episodios...")
            
            for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                hits_this_episode = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    done = done or truncated
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    if info.get('hit_puck', False):
                        hits_this_episode += 1
                
                # Recopilar estadísticas del episodio
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                goals_scored.append(info.get('ai_score', 0))
                goals_conceded.append(info.get('player_score', 0))
                hit_counts.append(hits_this_episode)
                
                # Determinar si ganó
                ai_score = info.get('ai_score', 0)
                player_score = info.get('player_score', 0)
                win_rates.append(1 if ai_score > player_score else 0)
                
                if (episode + 1) % 20 == 0:
                    print(f"  Episodio {episode + 1}/{n_episodes} completado")
            
            # Calcular estadísticas
            stats = {
                'model_name': model_name,
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'win_rate': np.mean(win_rates),
                'avg_goals_scored': np.mean(goals_scored),
                'avg_goals_conceded': np.mean(goals_conceded),
                'avg_hits_per_episode': np.mean(hit_counts),
                'goal_efficiency': np.mean(goals_scored) / max(np.mean(hit_counts), 1),
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'goals_scored': goals_scored,
                'goals_conceded': goals_conceded,
                'hit_counts': hit_counts,
                'win_rates': win_rates
            }
            
            self.results[model_name] = stats
            
            print(f"Resultados para {model_name}:")
            print(f"  Recompensa promedio: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"  Tasa de victoria: {stats['win_rate']:.2%}")
            print(f"  Goles promedio por episodio: {stats['avg_goals_scored']:.2f}")
            print(f"  Hits promedio por episodio: {stats['avg_hits_per_episode']:.2f}")
            print(f"  Eficiencia de gol: {stats['goal_efficiency']:.3f}")
            
            return stats
            
        except Exception as e:
            print(f"Error evaluando modelo {model_name}: {e}")
            return None
    
    def compare_models(self):
        """Comparar todos los modelos evaluados"""
        if len(self.results) < 2:
            print("Se necesitan al menos 2 modelos para comparar")
            return
        
        print("\n" + "="*60)
        print("COMPARACIÓN DE MODELOS")
        print("="*60)
        
        # Crear DataFrame para comparación
        comparison_data = []
        for model_name, stats in self.results.items():
            comparison_data.append({
                'Modelo': model_name,
                'Recompensa Media': stats['mean_reward'],
                'Tasa de Victoria': stats['win_rate'],
                'Goles por Episodio': stats['avg_goals_scored'],
                'Hits por Episodio': stats['avg_hits_per_episode'],
                'Eficiencia de Gol': stats['goal_efficiency'],
                'Duración Media': stats['mean_length']
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False, float_format='%.3f'))
        
        # Encontrar el mejor modelo
        best_model = max(self.results.keys(), 
                        key=lambda x: self.results[x]['win_rate'])
        
        print(f"\n🏆 MEJOR MODELO: {best_model}")
        print(f"   Tasa de victoria: {self.results[best_model]['win_rate']:.2%}")
        
        return df
    
    def plot_performance_comparison(self, save_path="model_comparison.png"):
        """Crear gráficos de comparación de rendimiento"""
        if len(self.results) < 2:
            print("Se necesitan al menos 2 modelos para crear gráficos")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comparación de Rendimiento de Modelos', fontsize=16)
        
        model_names = list(self.results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        # 1. Distribución de recompensas
        axes[0, 0].set_title('Distribución de Recompensas por Episodio')
        for i, (name, stats) in enumerate(self.results.items()):
            axes[0, 0].hist(stats['episode_rewards'], alpha=0.7, 
                           label=name, color=colors[i], bins=30)
        axes[0, 0].set_xlabel('Recompensa')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].legend()
        
        # 2. Tasa de victoria
        win_rates = [stats['win_rate'] for stats in self.results.values()]
        axes[0, 1].bar(model_names, win_rates, color=colors)
        axes[0, 1].set_title('Tasa de Victoria')
        axes[0, 1].set_ylabel('Tasa de Victoria')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(win_rates):
            axes[0, 1].text(i, v + 0.02, f'{v:.2%}', ha='center')
        
        # 3. Goles por episodio
        goals_scored = [stats['avg_goals_scored'] for stats in self.results.values()]
        goals_conceded = [stats['avg_goals_conceded'] for stats in self.results.values()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 2].bar(x - width/2, goals_scored, width, label='Goles Anotados', color='green', alpha=0.7)
        axes[0, 2].bar(x + width/2, goals_conceded, width, label='Goles Recibidos', color='red', alpha=0.7)
        axes[0, 2].set_title('Goles por Episodio')
        axes[0, 2].set_ylabel('Goles')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(model_names)
        axes[0, 2].legend()
        
        # 4. Evolución de recompensas (primeros 50 episodios)
        axes[1, 0].set_title('Evolución de Recompensas (Primeros 50 Episodios)')
        for name, stats in self.results.items():
            rewards_subset = stats['episode_rewards'][:50]
            axes[1, 0].plot(rewards_subset, label=name, alpha=0.8)
        axes[1, 0].set_xlabel('Episodio')
        axes[1, 0].set_ylabel('Recompensa')
        axes[1, 0].legend()
        
        # 5. Eficiencia de gol vs Hits
        axes[1, 1].set_title('Eficiencia de Gol vs Hits por Episodio')
        for name, stats in self.results.items():
            axes[1, 1].scatter(stats['avg_hits_per_episode'], 
                              stats['goal_efficiency'], 
                              label=name, s=100, alpha=0.7)
        axes[1, 1].set_xlabel('Hits Promedio por Episodio')
        axes[1, 1].set_ylabel('Eficiencia de Gol')
        axes[1, 1].legend()
        
        # 6. Duración de episodios
        axes[1, 2].set_title('Duración de Episodios')
        episode_lengths = [stats['episode_lengths'] for stats in self.results.values()]
        axes[1, 2].boxplot(episode_lengths, labels=model_names)
        axes[1, 2].set_ylabel('Duración (pasos)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráficos guardados en: {save_path}")
        plt.show()
    
    def analyze_learning_progression(self, model_name):
        """Analizar la progresión del aprendizaje durante un episodio"""
        if model_name not in self.results:
            print(f"Modelo {model_name} no encontrado")
            return
        
        stats = self.results[model_name]
        
        # Análisis de progresión
        rewards = stats['episode_rewards']
        window_size = 10
        
        # Calcular media móvil
        moving_avg = []
        for i in range(len(rewards)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(np.mean(rewards[start_idx:i+1]))
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(rewards, alpha=0.6, label='Recompensa por episodio')
        plt.plot(moving_avg, label=f'Media móvil (ventana={window_size})', linewidth=2)
        plt.title(f'Progresión del Aprendizaje - {model_name}')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(stats['win_rates'], alpha=0.7)
        plt.title('Victorias por Episodio')
        plt.xlabel('Episodio')
        plt.ylabel('Victoria (1) / Derrota (0)')
        
        plt.subplot(2, 2, 3)
        plt.plot(stats['goals_scored'], label='Goles Anotados', alpha=0.7)
        plt.plot(stats['goals_conceded'], label='Goles Recibidos', alpha=0.7)
        plt.title('Goles por Episodio')
        plt.xlabel('Episodio')
        plt.ylabel('Goles')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(stats['hit_counts'], alpha=0.7)
        plt.title('Hits por Episodio')
        plt.xlabel('Episodio')
        plt.ylabel('Número de Hits')
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_learning_progression.png', dpi=300, bbox_inches='tight')
        plt.show()

def test_different_difficulties():
    """Probar un modelo contra diferentes niveles de dificultad"""
    print("Probando modelo contra diferentes niveles de dificultad...")
    
    # Buscar modelo mejorado
    model_path = None
    if os.path.exists("../improved_models/improved_air_hockey_final.zip"):
        model_path = "../improved_models/improved_air_hockey_final.zip"
    elif os.path.exists("../models/air_hockey_ppo_final.zip"):
        model_path = "../models/air_hockey_ppo_final.zip"
    else:
        print("No se encontró ningún modelo entrenado")
        return
    
    model = PPO.load(model_path)
    
    difficulty_results = {}
    
    for difficulty in range(6):
        print(f"\nProbando dificultad nivel {difficulty}...")
        
        env = ImprovedAirHockeyEnv()
        env.set_difficulty_level(difficulty)
        
        wins = 0
        total_episodes = 20
        total_reward = 0
        
        for episode in range(total_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                done = done or truncated
                episode_reward += reward
            
            total_reward += episode_reward
            if info.get('ai_score', 0) > info.get('player_score', 0):
                wins += 1
        
        win_rate = wins / total_episodes
        avg_reward = total_reward / total_episodes
        
        difficulty_results[difficulty] = {
            'win_rate': win_rate,
            'avg_reward': avg_reward
        }
        
        print(f"  Tasa de victoria: {win_rate:.2%}")
        print(f"  Recompensa promedio: {avg_reward:.2f}")
    
    # Graficar resultados
    difficulties = list(difficulty_results.keys())
    win_rates = [difficulty_results[d]['win_rate'] for d in difficulties]
    avg_rewards = [difficulty_results[d]['avg_reward'] for d in difficulties]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(difficulties, win_rates, 'bo-', linewidth=2, markersize=8)
    ax1.set_title('Tasa de Victoria vs Dificultad')
    ax1.set_xlabel('Nivel de Dificultad')
    ax1.set_ylabel('Tasa de Victoria')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(difficulties, avg_rewards, 'ro-', linewidth=2, markersize=8)
    ax2.set_title('Recompensa Promedio vs Dificultad')
    ax2.set_xlabel('Nivel de Dificultad')
    ax2.set_ylabel('Recompensa Promedio')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('difficulty_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return difficulty_results

def main():
    """Función principal para análisis de modelos"""
    print("=== ANÁLISIS DE RENDIMIENTO DE MODELOS ===")
    
    analyzer = ModelAnalyzer()
    
    # Buscar modelos disponibles
    models_to_test = []
    
    # Modelo mejorado
    if os.path.exists("../improved_models/improved_air_hockey_final.zip"):
        models_to_test.append(("improved_models/improved_air_hockey_final.zip", "Modelo Mejorado", "improved"))
    
    # Modelo original
    if os.path.exists("../models/air_hockey_ppo_final.zip"):
        models_to_test.append(("models/air_hockey_ppo_final.zip", "Modelo Original", "original"))
    
    # Modelo DQN si existe
    if os.path.exists("../air_hockey_dqn.zip"):
        models_to_test.append(("air_hockey_dqn.zip", "Modelo DQN", "original"))
    
    if not models_to_test:
        print("No se encontraron modelos entrenados para analizar")
        print("Entrena un modelo primero usando train_agent.py o improved_training_system.py")
        return
    
    print(f"Encontrados {len(models_to_test)} modelos para analizar")
    
    # Evaluar cada modelo
    for model_path, model_name, env_type in models_to_test:
        analyzer.evaluate_model_comprehensive(model_path, model_name, env_type, n_episodes=50)
    
    # Comparar modelos
    if len(analyzer.results) > 1:
        comparison_df = analyzer.compare_models()
        analyzer.plot_performance_comparison()
        
        # Guardar resultados en CSV
        comparison_df.to_csv('model_comparison_results.csv', index=False)
        print("\nResultados guardados en: model_comparison_results.csv")
    
    # Análisis de dificultad
    print("\n" + "="*60)
    choice = input("¿Realizar análisis de dificultad? (y/n): ").lower().startswith('y')
    if choice:
        test_different_difficulties()
    
    # Análisis de progresión de aprendizaje
    if analyzer.results:
        print("\n" + "="*60)
        print("Modelos disponibles para análisis de progresión:")
        for i, model_name in enumerate(analyzer.results.keys()):
            print(f"{i+1}. {model_name}")
        
        try:
            choice = input("Selecciona un modelo para análisis de progresión (número): ").strip()
            if choice:
                model_idx = int(choice) - 1
                model_names = list(analyzer.results.keys())
                if 0 <= model_idx < len(model_names):
                    analyzer.analyze_learning_progression(model_names[model_idx])
        except (ValueError, IndexError):
            print("Selección inválida")

if __name__ == "__main__":
    main() 