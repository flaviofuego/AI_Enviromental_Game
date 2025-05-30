# analyze_fixed_model_direct.py
from analyze_fixed_model import FixedModelAnalyzer
import os

# Ruta al modelo fixed
model_path = "improved_models/quick_fixed_model_final.zip"

if os.path.exists(model_path):
    print("🔬 INICIANDO ANÁLISIS COMPLETO DEL MODELO FIXED")
    print("="*70)
    print(f"Modelo: {model_path}")
    print("="*70)
    
    # Crear analizador
    analyzer = FixedModelAnalyzer(model_path)
    
    # Ejecutar análisis completo con 100 episodios
    print("\n⏳ Ejecutando 100 episodios de prueba...")
    print("Esto puede tomar varios minutos...\n")
    
    metrics = analyzer.analyze(num_episodes=100, verbose=True)
    
    print("\n✅ ANÁLISIS COMPLETADO")
    print("\n📊 RESUMEN DE MÉTRICAS CLAVE:")
    print("-"*50)
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    
    # Calcular movimiento vertical
    total_actions = sum(metrics['actions'].values())
    vertical_pct = (metrics['actions'][0] + metrics['actions'][1]) / total_actions * 100
    horizontal_pct = (metrics['actions'][2] + metrics['actions'][3]) / total_actions * 100
    stay_pct = metrics['actions'][4] / total_actions * 100
    
    print(f"Movimiento Vertical: {vertical_pct:.1f}%")
    print(f"Movimiento Horizontal: {horizontal_pct:.1f}%")
    print(f"Sin movimiento (Stay): {stay_pct:.1f}%")
    
    # Información adicional
    import numpy as np
    print(f"\nRecompensa promedio: {np.mean(metrics['rewards']):.2f}")
    print(f"Hits por episodio: {np.mean(metrics['hits']):.2f}")
    print(f"Goles anotados promedio: {np.mean(metrics['goals_scored']):.2f}")
    print(f"Goles recibidos promedio: {np.mean(metrics['goals_conceded']):.2f}")
    
    print("\n📁 Los resultados detallados se han guardado en la carpeta analysis_results_*")
else:
    print(f"❌ No se encontró el modelo en: {model_path}") 