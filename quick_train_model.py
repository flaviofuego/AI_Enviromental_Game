#!/usr/bin/env python3
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
    print(f"   Modelo guardado como: models/{args.name}_final.zip")

if __name__ == "__main__":
    main()
