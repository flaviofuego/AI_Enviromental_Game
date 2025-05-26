# setup_improved_training.py
import os
import subprocess
import sys

def install_requirements():
    """Instalar dependencias necesarias para el sistema mejorado"""
    print("🔧 Instalando dependencias...")
    
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
            print(f"✅ {package} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"❌ Error instalando {package}")
            return False
    
    return True

def create_directories():
    """Crear estructura de directorios necesaria"""
    print("\n📁 Creando estructura de directorios...")
    
    directories = [
        "improved_models",
        "improved_models/best_model",
        "improved_logs", 
        "analysis_results",
        "checkpoints"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directorio creado: {directory}")

def check_existing_files():
    """Verificar archivos existentes del proyecto"""
    print("\n🔍 Verificando archivos del proyecto...")
    
    required_files = [
        "constants.py",
        "sprites.py", 
        "table.py",
        "utils.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} encontrado")
        else:
            print(f"❌ {file} no encontrado")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Archivos faltantes: {missing_files}")
        print("Asegúrate de tener todos los archivos del proyecto original")
        return False
    
    return True

def run_quick_test():
    """Ejecutar una prueba rápida del sistema"""
    print("\n🧪 Ejecutando prueba rápida...")
    
    try:
        # Importar el sistema mejorado
        from improved_training_system import ImprovedAirHockeyEnv
        
        # Crear entorno de prueba
        env = ImprovedAirHockeyEnv()
        obs, _ = env.reset()
        
        print(f"✅ Entorno creado correctamente")
        print(f"✅ Observación shape: {obs.shape}")
        print(f"✅ Action space: {env.action_space}")
        
        # Probar un paso
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"✅ Step ejecutado correctamente")
        print(f"✅ Reward: {reward}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

def create_training_script():
    """Crear script de entrenamiento rápido"""
    print("\n📝 Creando script de entrenamiento rápido...")
    
    script_content = '''#!/usr/bin/env python3
# quick_train.py - Script de entrenamiento rápido

from improved_training_system import train_improved_agent
import argparse

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento rápido del modelo mejorado')
    parser.add_argument('--timesteps', type=int, default=500000, 
                       help='Número de timesteps para entrenar (default: 500000)')
    parser.add_argument('--name', type=str, default='quick_model',
                       help='Nombre del modelo (default: quick_model)')
    
    args = parser.parse_args()
    
    print(f"🚀 Iniciando entrenamiento rápido...")
    print(f"   Timesteps: {args.timesteps}")
    print(f"   Nombre: {args.name}")
    
    model = train_improved_agent(
        total_timesteps=args.timesteps,
        model_name=args.name
    )
    
    print(f"✅ Entrenamiento completado!")
    print(f"   Modelo guardado como: improved_models/{args.name}_final.zip")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_train.py", "w", encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ Script quick_train.py creado")

def create_analysis_script():
    """Crear script de análisis rápido"""
    print("\n📊 Creando script de análisis rápido...")
    
    script_content = '''#!/usr/bin/env python3
# quick_analysis.py - Análisis rápido de modelos

from training_analysis import ModelAnalyzer
import os

def main():
    print("🔍 Análisis rápido de modelos...")
    
    analyzer = ModelAnalyzer()
    
    # Buscar modelos
    models_found = []
    
    if os.path.exists("improved_models/quick_model_final.zip"):
        models_found.append(("improved_models/quick_model_final.zip", "Modelo Rápido", "improved"))
    
    if os.path.exists("models/air_hockey_ppo_final.zip"):
        models_found.append(("models/air_hockey_ppo_final.zip", "Modelo Original", "original"))
    
    if not models_found:
        print("❌ No se encontraron modelos para analizar")
        print("   Entrena un modelo primero con: python quick_train.py")
        return
    
    # Evaluar modelos encontrados
    for model_path, model_name, env_type in models_found:
        print(f"\n📈 Evaluando {model_name}...")
        analyzer.evaluate_model_comprehensive(model_path, model_name, env_type, n_episodes=20)
    
    # Comparar si hay múltiples modelos
    if len(analyzer.results) > 1:
        analyzer.compare_models()
        analyzer.plot_performance_comparison("quick_analysis.png")
        print("\n✅ Análisis completado. Gráficos guardados en quick_analysis.png")
    else:
        print("\n✅ Evaluación completada")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_analysis.py", "w", encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ Script quick_analysis.py creado")

def print_usage_instructions():
    """Imprimir instrucciones de uso"""
    print("\n" + "="*60)
    print("🎉 CONFIGURACIÓN COMPLETADA")
    print("="*60)
    
    print("\n🚀 CÓMO USAR EL SISTEMA MEJORADO:")
    print("\n1. Entrenamiento rápido (500k pasos):")
    print("   python quick_train.py")
    
    print("\n2. Entrenamiento completo (2M pasos):")
    print("   python improved_training_system.py")
    
    print("\n3. Análisis de modelos:")
    print("   python quick_analysis.py")
    print("   python training_analysis.py")
    
    print("\n4. Entrenamiento personalizado:")
    print("   python quick_train.py --timesteps 1000000 --name mi_modelo")
    
    print("\n📊 ARCHIVOS IMPORTANTES:")
    print("   - improved_training_system.py: Sistema completo mejorado")
    print("   - training_analysis.py: Análisis detallado de rendimiento")
    print("   - quick_train.py: Entrenamiento rápido")
    print("   - quick_analysis.py: Análisis rápido")
    print("   - README_MEJORAS_ENTRENAMIENTO.md: Documentación completa")
    
    print("\n💡 CONSEJOS:")
    print("   - Empieza con quick_train.py para probar el sistema")
    print("   - Usa training_analysis.py para comparar modelos")
    print("   - Lee README_MEJORAS_ENTRENAMIENTO.md para detalles")
    print("   - Los modelos se guardan en improved_models/")

def main():
    """Función principal de configuración"""
    print("🏒 CONFIGURACIÓN DEL SISTEMA DE ENTRENAMIENTO MEJORADO")
    print("="*60)
    
    # Verificar archivos existentes
    if not check_existing_files():
        print("\n❌ Configuración abortada. Archivos faltantes.")
        return
    
    # Instalar dependencias
    if not install_requirements():
        print("\n❌ Error instalando dependencias.")
        return
    
    # Crear directorios
    create_directories()
    
    # Crear scripts auxiliares
    create_training_script()
    create_analysis_script()
    
    # Ejecutar prueba rápida
    if not run_quick_test():
        print("\n⚠️  Prueba rápida falló. Revisa las dependencias.")
        return
    
    # Mostrar instrucciones
    print_usage_instructions()

if __name__ == "__main__":
    main() 