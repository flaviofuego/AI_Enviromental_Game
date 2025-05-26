#!/usr/bin/env python3
# quick_analysis.py - Analisis rapido de modelos

from training_analysis import ModelAnalyzer
import os

def main():
    print("Analisis rapido de modelos...")
    
    analyzer = ModelAnalyzer()
    
    # Buscar modelos
    models_found = []
    
    if os.path.exists("improved_models/quick_model_final.zip"):
        models_found.append(("improved_models/quick_model_final.zip", "Modelo Rapido", "improved"))
    
    if os.path.exists("models/air_hockey_ppo_final.zip"):
        models_found.append(("models/air_hockey_ppo_final.zip", "Modelo Original", "original"))
    
    if not models_found:
        print("ERROR: No se encontraron modelos para analizar")
        print("   Entrena un modelo primero con: python quick_train.py")
        return
    
    # Evaluar modelos encontrados
    for model_path, model_name, env_type in models_found:
        print(f"\nEvaluando {model_name}...")
        analyzer.evaluate_model_comprehensive(model_path, model_name, env_type, n_episodes=20)
    
    # Comparar si hay multiples modelos
    if len(analyzer.results) > 1:
        analyzer.compare_models()
        analyzer.plot_performance_comparison("quick_analysis.png")
        print("\nAnalisis completado. Graficos guardados en quick_analysis.png")
    else:
        print("\nEvaluacion completada")

if __name__ == "__main__":
    main()
