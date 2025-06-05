# setup_simple.py
import os
import subprocess
import sys

def install_requirements():
    """Instalar dependencias necesarias para el sistema mejorado"""
    print("Instalando dependencias...")
    
    requirements = [
        "stable-baselines3[extra]",
        "gymnasium",
        "pygame", 
        "numpy",
        "torch",
        "matplotlib",
        "seaborn",
        "pandas",
        "tensorboard"
    ]
    
    for package in requirements:
        try:
            print(f"Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"OK: {package} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"ERROR: Error instalando {package}")
            return False
    
    return True

def create_directories():
    """Crear estructura de directorios necesaria"""
    print("\nCreando estructura de directorios...")
    
    directories = [
        "improved_models",
        "improved_models/best_model",
        "improved_logs", 
        "analysis_results",
        "checkpoints"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"OK: Directorio creado: {directory}")

def check_existing_files():
    """Verificar archivos existentes del proyecto"""
    print("\nVerificando archivos del proyecto...")
    
    required_files = [
        "constants.py",
        "sprites.py", 
        "table.py",
        "utils.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"OK: {file} encontrado")
        else:
            print(f"ERROR: {file} no encontrado")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nArchivos faltantes: {missing_files}")
        print("Asegurate de tener todos los archivos del proyecto original")
        return False
    
    return True

def run_quick_test():
    """Ejecutar una prueba rapida del sistema"""
    print("\nEjecutando prueba rapida...")
    
    try:
        # Importar el sistema mejorado
        from training_Systems.improved_training_system import ImprovedAirHockeyEnv
        
        # Crear entorno de prueba
        env = ImprovedAirHockeyEnv()
        obs, _ = env.reset()
        
        print(f"OK: Entorno creado correctamente")
        print(f"OK: Observacion shape: {obs.shape}")
        print(f"OK: Action space: {env.action_space}")
        
        # Probar un paso
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"OK: Step ejecutado correctamente")
        print(f"OK: Reward: {reward}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"ERROR: Error en prueba: {e}")
        return False

def create_training_script():
    """Crear script de entrenamiento rapido"""
    print("\nCreando script de entrenamiento rapido...")
    
    script_content = '''#!/usr/bin/env python3
# quick_train.py - Script de entrenamiento rapido

from improved_training_system import train_improved_agent
import argparse

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento rapido del modelo mejorado')
    parser.add_argument('--timesteps', type=int, default=500000, 
                       help='Numero de timesteps para entrenar (default: 500000)')
    parser.add_argument('--name', type=str, default='quick_model',
                       help='Nombre del modelo (default: quick_model)')
    
    args = parser.parse_args()
    
    print(f"Iniciando entrenamiento rapido...")
    print(f"   Timesteps: {args.timesteps}")
    print(f"   Nombre: {args.name}")
    
    model = train_improved_agent(
        total_timesteps=args.timesteps,
        model_name=args.name
    )
    
    print(f"Entrenamiento completado!")
    print(f"   Modelo guardado como: improved_models/{args.name}_final.zip")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_train.py", "w", encoding='utf-8') as f:
        f.write(script_content)
    
    print("OK: Script quick_train.py creado")

def create_analysis_script():
    """Crear script de analisis rapido"""
    print("\nCreando script de analisis rapido...")
    
    script_content = '''#!/usr/bin/env python3
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
        print(f"\\nEvaluando {model_name}...")
        analyzer.evaluate_model_comprehensive(model_path, model_name, env_type, n_episodes=20)
    
    # Comparar si hay multiples modelos
    if len(analyzer.results) > 1:
        analyzer.compare_models()
        analyzer.plot_performance_comparison("quick_analysis.png")
        print("\\nAnalisis completado. Graficos guardados en quick_analysis.png")
    else:
        print("\\nEvaluacion completada")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_analysis.py", "w", encoding='utf-8') as f:
        f.write(script_content)
    
    print("OK: Script quick_analysis.py creado")

def print_usage_instructions():
    """Imprimir instrucciones de uso"""
    print("\n" + "="*60)
    print("CONFIGURACION COMPLETADA")
    print("="*60)
    
    print("\nCOMO USAR EL SISTEMA MEJORADO:")
    print("\n1. Entrenamiento rapido (500k pasos):")
    print("   python quick_train.py")
    
    print("\n2. Entrenamiento completo (2M pasos):")
    print("   python improved_training_system.py")
    
    print("\n3. Analisis de modelos:")
    print("   python quick_analysis.py")
    print("   python training_analysis.py")
    
    print("\n4. Entrenamiento personalizado:")
    print("   python quick_train.py --timesteps 1000000 --name mi_modelo")
    
    print("\nARCHIVOS IMPORTANTES:")
    print("   - improved_training_system.py: Sistema completo mejorado")
    print("   - training_analysis.py: Analisis detallado de rendimiento")
    print("   - quick_train.py: Entrenamiento rapido")
    print("   - quick_analysis.py: Analisis rapido")
    print("   - README_MEJORAS_ENTRENAMIENTO.md: Documentacion completa")
    
    print("\nCONSEJOS:")
    print("   - Empieza con quick_train.py para probar el sistema")
    print("   - Usa training_analysis.py para comparar modelos")
    print("   - Lee README_MEJORAS_ENTRENAMIENTO.md para detalles")
    print("   - Los modelos se guardan en improved_models/")

def main():
    """Funcion principal de configuracion"""
    print("CONFIGURACION DEL SISTEMA DE ENTRENAMIENTO MEJORADO")
    print("="*60)
    
    # Verificar archivos existentes
    if not check_existing_files():
        print("\nConfiguracion abortada. Archivos faltantes.")
        return
    
    # Instalar dependencias
    if not install_requirements():
        print("\nError instalando dependencias.")
        return
    
    # Crear directorios
    create_directories()
    
    # Crear scripts auxiliares
    create_training_script()
    create_analysis_script()
    
    # Ejecutar prueba rapida
    if not run_quick_test():
        print("\nPrueba rapida fallo. Revisa las dependencias.")
        return
    
    # Mostrar instrucciones
    print_usage_instructions()

if __name__ == "__main__":
    main() 