# quick_diagnosis.py
import os
import numpy as np
from stable_baselines3 import PPO
from improved_training_system_fixed import FixedAirHockeyEnv, test_fixed_model_behavior

def find_all_models():
    """Encuentra todos los modelos disponibles"""
    models = []
    
    # Buscar en diferentes directorios
    dirs_to_search = ["improved_models", "models", "."]
    
    for directory in dirs_to_search:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith((".zip", ".pkl")):
                    model_path = os.path.join(directory, file)
                    models.append(model_path)
    
    return models

def quick_test_model(model_path, num_tests=500):
    """Prueba rápida de un modelo"""
    print(f"\n🔍 QUICK TEST: {model_path}")
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None
    
    env = FixedAirHockeyEnv()
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    obs, _ = env.reset()
    
    for i in range(num_tests):
        try:
            action, _ = model.predict(obs, deterministic=True)
            
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.ndim == 0 else int(action[0])
            else:
                action = int(action)
            
            action_counts[action] += 1
            
            obs, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                obs, _ = env.reset()
                
        except Exception as e:
            print(f"Error en step {i}: {e}")
            obs, _ = env.reset()
    
    # Análisis rápido
    total = sum(action_counts.values())
    if total == 0:
        print("❌ CRÍTICO: No se pudieron ejecutar acciones")
        return None
    
    vertical_pct = ((action_counts[0] + action_counts[1]) / total) * 100
    horizontal_pct = ((action_counts[2] + action_counts[3]) / total) * 100
    stay_pct = (action_counts[4] / total) * 100
    
    print(f"  Vertical: {vertical_pct:.1f}% | Horizontal: {horizontal_pct:.1f}% | Stay: {stay_pct:.1f}%")
    
    # Evaluación
    if vertical_pct == 0:
        status = "❌ SIN MOVIMIENTO VERTICAL"
    elif vertical_pct < 10:
        status = "⚠️  POCO MOVIMIENTO VERTICAL"
    elif vertical_pct < 25:
        status = "📈 MOVIMIENTO MODERADO"
    else:
        status = "✅ BUEN BALANCE"
    
    print(f"  {status}")
    return {"vertical": vertical_pct, "horizontal": horizontal_pct, "stay": stay_pct, "status": status}

def main():
    print("🚀 DIAGNÓSTICO RÁPIDO DE MODELOS")
    print("=" * 50)
    
    models = find_all_models()
    
    if not models:
        print("❌ No se encontraron modelos para probar")
        return
    
    print(f"Encontrados {len(models)} modelos:")
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    
    print("\nOpciones:")
    print("0. Probar todos los modelos (quick test)")
    print("A. Análisis completo de un modelo específico")
    
    choice = input("\nSelecciona opción (0/A/1-{}): ".format(len(models))).strip()
    
    if choice == "0":
        print("\n🔍 PROBANDO TODOS LOS MODELOS...")
        results = []
        
        for model in models:
            result = quick_test_model(model)
            if result:
                results.append((model, result))
        
        # Ranking
        print(f"\n{'='*60}")
        print("RANKING DE MODELOS")
        print(f"{'='*60}")
        
        # Ordenar por movimiento vertical
        results.sort(key=lambda x: x[1]["vertical"], reverse=True)
        
        for i, (model, result) in enumerate(results):
            print(f"{i+1}. {os.path.basename(model)}")
            print(f"   Vertical: {result['vertical']:.1f}% | {result['status']}")
    
    elif choice.upper() == "A":
        print("\nSelecciona modelo para análisis completo:")
        for i, model in enumerate(models):
            print(f"{i+1}. {model}")
        
        try:
            idx = int(input(f"Modelo (1-{len(models)}): ")) - 1
            if 0 <= idx < len(models):
                print(f"\n📊 ANÁLISIS COMPLETO DE: {models[idx]}")
                test_fixed_model_behavior(models[idx], num_tests=2000)
        except ValueError:
            print("❌ Entrada inválida")
    
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                quick_test_model(models[idx])
        except ValueError:
            print("❌ Entrada inválida")

if __name__ == "__main__":
    main() 