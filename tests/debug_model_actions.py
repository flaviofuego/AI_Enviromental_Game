# debug_model_actions.py
import numpy as np
import pygame
import sys
import os
from stable_baselines3 import PPO
from collections import Counter
import time

# Import game components
from constants import *
from sprites import Puck, HumanMallet, AIMallet
from table import Table

def create_observation_for_model(ai_mallet, puck, human_mallet, player_score, ai_score, model_type="original"):
    """Create observation vector based on model type"""
    
    if model_type == "improved":
        # Enhanced observation for improved models (21 dimensions)
        ai_x_norm = ai_mallet.position[0] / WIDTH
        ai_y_norm = ai_mallet.position[1] / HEIGHT
        puck_x_norm = puck.position[0] / WIDTH
        puck_y_norm = puck.position[1] / HEIGHT
        human_x_norm = human_mallet.position[0] / WIDTH
        human_y_norm = human_mallet.position[1] / HEIGHT
        
        # Normalizar velocidades
        max_velocity = 20.0
        puck_vx_norm = np.clip(puck.velocity[0] / max_velocity, -1, 1)
        puck_vy_norm = np.clip(puck.velocity[1] / max_velocity, -1, 1)
        ai_vx_norm = np.clip(ai_mallet.velocity[0] / max_velocity, -1, 1)
        ai_vy_norm = np.clip(ai_mallet.velocity[1] / max_velocity, -1, 1)
        human_vx_norm = np.clip(human_mallet.velocity[0] / max_velocity, -1, 1)
        human_vy_norm = np.clip(human_mallet.velocity[1] / max_velocity, -1, 1)
        
        # Distancias normalizadas
        max_distance = np.sqrt(WIDTH**2 + HEIGHT**2)
        puck_to_ai_dist = np.sqrt(
            (puck.position[0] - ai_mallet.position[0])**2 + 
            (puck.position[1] - ai_mallet.position[1])**2
        ) / max_distance
        puck_to_human_dist = np.sqrt(
            (puck.position[0] - human_mallet.position[0])**2 + 
            (puck.position[1] - human_mallet.position[1])**2
        ) / max_distance
        
        # Información contextual
        puck_in_ai_half = 1.0 if puck.position[0] > WIDTH // 2 else -1.0
        puck_moving_to_ai_goal = 1.0 if puck.velocity[0] > 0 else -1.0
        puck_moving_to_human_goal = 1.0 if puck.velocity[0] < 0 else -1.0
        
        # Información de tiempo y estado
        time_factor = 0.5
        score_diff = (ai_score - player_score) / 7.0
        
        # Predicción de trayectoria del puck
        if abs(puck.velocity[0]) > 0.1:
            time_to_ai_side = (WIDTH - puck.position[0]) / puck.velocity[0] if puck.velocity[0] > 0 else 0
            predicted_y_at_ai_side = puck.position[1] + puck.velocity[1] * time_to_ai_side
            predicted_y_norm = np.clip(predicted_y_at_ai_side / HEIGHT, 0, 1)
        else:
            predicted_y_norm = puck_y_norm
        
        # Construir vector de observación (21 dimensiones)
        observation = np.array([
            ai_x_norm, ai_y_norm, puck_x_norm, puck_y_norm, human_x_norm, human_y_norm,
            puck_vx_norm, puck_vy_norm, ai_vx_norm, ai_vy_norm, human_vx_norm, human_vy_norm,
            puck_to_ai_dist, puck_to_human_dist,
            puck_in_ai_half, puck_moving_to_ai_goal, puck_moving_to_human_goal,
            time_factor, score_diff, predicted_y_norm, 0.5
        ], dtype=np.float32)
        
    else:
        # Original observation for legacy models (13 dimensions)
        basic_obs = np.array([
            ai_mallet.position[0] / WIDTH,
            ai_mallet.position[1] / HEIGHT,
            puck.position[0] / WIDTH,
            puck.position[1] / HEIGHT,
            np.clip(puck.velocity[0] / puck.max_speed, -1, 1),
            np.clip(puck.velocity[1] / puck.max_speed, -1, 1)
        ], dtype=np.float32)
        
        puck_to_mallet_dist = np.sqrt(
            (puck.position[0] - ai_mallet.position[0])**2 + 
            (puck.position[1] - ai_mallet.position[1])**2
        ) / np.sqrt(WIDTH**2 + HEIGHT**2)
        
        puck_to_ai_goal = (WIDTH - puck.position[0]) / WIDTH
        puck_to_player_goal = puck.position[0] / WIDTH
        time_since_hit = 0.5
        puck_moving_to_player = 1.0 if puck.velocity[0] < 0 else 0.0
        
        observation = np.append(basic_obs, [
            puck_to_mallet_dist, puck_to_ai_goal, puck_to_player_goal,
            time_since_hit, puck_moving_to_player, player_score / 5.0, ai_score / 5.0
        ])
    
    return observation

def debug_model_actions(model_path, num_steps=1000):
    """Debug what actions the model is taking"""
    print(f"Debugging model: {model_path}")
    
    # Load model
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Determine model type
    obs_space = model.observation_space
    obs_dim = obs_space.shape[0]
    if obs_dim == 21:
        model_type = "improved"
    elif obs_dim == 13:
        model_type = "original"
    else:
        model_type = f"unknown ({obs_dim} dims)"
    
    print(f"Model type: {model_type}")
    print(f"Observation dimensions: {obs_dim}")
    
    # Initialize pygame (minimal)
    pygame.init()
    
    # Create game objects
    table = Table()
    human_mallet = HumanMallet()
    puck = Puck()
    ai_mallet = AIMallet()
    
    # Reset positions
    human_mallet.position = [WIDTH // 4, HEIGHT // 2]
    ai_mallet.position = [WIDTH * 3 // 4, HEIGHT // 2]
    puck.position = [WIDTH // 2, HEIGHT // 2]
    puck.velocity = [5, 3]  # Give it some initial velocity
    
    # Track actions
    action_counter = Counter()
    action_names = ["Up", "Down", "Left", "Right", "Stay"]
    
    print(f"\nRunning {num_steps} prediction steps...")
    
    for step in range(num_steps):
        # Create observation
        observation = create_observation_for_model(
            ai_mallet, puck, human_mallet, 0, 0, model_type
        )
        
        # Get action from model
        try:
            action, _ = model.predict(observation, deterministic=True)
            # Convert numpy array to int if necessary
            if isinstance(action, np.ndarray):
                if action.ndim == 0:  # 0-dimensional array
                    action = int(action.item())
                else:
                    action = int(action[0])
            else:
                action = int(action)
            action_counter[action] += 1
            
            # Print some debug info every 100 steps
            if step % 100 == 0:
                print(f"Step {step}: Action = {action} ({action_names[action]})")
                print(f"  AI Mallet pos: ({ai_mallet.position[0]:.1f}, {ai_mallet.position[1]:.1f})")
                print(f"  Puck pos: ({puck.position[0]:.1f}, {puck.position[1]:.1f})")
                print(f"  Observation sample: {observation[:6]}")
        
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
        
        # Simulate some movement to change the observation
        # Move puck slightly
        puck.position[0] += puck.velocity[0] * 0.1
        puck.position[1] += puck.velocity[1] * 0.1
        
        # Keep puck in bounds
        if puck.position[0] < 0 or puck.position[0] > WIDTH:
            puck.velocity[0] *= -1
        if puck.position[1] < 0 or puck.position[1] > HEIGHT:
            puck.velocity[1] *= -1
        
        puck.position[0] = max(0, min(WIDTH, puck.position[0]))
        puck.position[1] = max(0, min(HEIGHT, puck.position[1]))
        
        # Move human mallet slightly (simulate human movement)
        human_mallet.position[0] += np.random.normal(0, 1)
        human_mallet.position[1] += np.random.normal(0, 1)
        human_mallet.position[0] = max(30, min(WIDTH//2 - 30, human_mallet.position[0]))
        human_mallet.position[1] = max(30, min(HEIGHT - 30, human_mallet.position[1]))
        
        # Apply the action to AI mallet
        move_amount = 7
        if action == 0:  # Up
            ai_mallet.position[1] = max(ai_mallet.position[1] - move_amount, 30)
        elif action == 1:  # Down
            ai_mallet.position[1] = min(ai_mallet.position[1] + move_amount, HEIGHT - 30)
        elif action == 2:  # Left
            ai_mallet.position[0] = max(ai_mallet.position[0] - move_amount, WIDTH // 2 + 30)
        elif action == 3:  # Right
            ai_mallet.position[0] = min(ai_mallet.position[0] + move_amount, WIDTH - 30)
        # action == 4 is "stay"
    
    pygame.quit()
    
    # Print results
    print(f"\n{'='*50}")
    print("ACTION DISTRIBUTION")
    print(f"{'='*50}")
    
    total_actions = sum(action_counter.values())
    for action_id in range(5):
        count = action_counter[action_id]
        percentage = (count / total_actions) * 100 if total_actions > 0 else 0
        print(f"{action_names[action_id]:>6}: {count:>4} ({percentage:>5.1f}%)")
    
    print(f"\nTotal actions: {total_actions}")
    
    # Analysis
    print(f"\n{'='*50}")
    print("ANALYSIS")
    print(f"{'='*50}")
    
    if action_counter[4] > total_actions * 0.8:
        print("⚠️  WARNING: Model is mostly staying still (>80% Stay actions)")
        print("   This suggests the model hasn't learned proper movement")
    
    vertical_actions = action_counter[0] + action_counter[1]  # Up + Down
    horizontal_actions = action_counter[2] + action_counter[3]  # Left + Right
    
    if total_actions > 0:
        print(f"Vertical movement (Up/Down): {vertical_actions} ({(vertical_actions/total_actions)*100:.1f}%)")
        print(f"Horizontal movement (Left/Right): {horizontal_actions} ({(horizontal_actions/total_actions)*100:.1f}%)")
    else:
        print("No actions recorded - model prediction failed")
    
    if vertical_actions == 0:
        print("❌ PROBLEM: No vertical movement detected!")
        print("   The model is not learning to move up/down")
    elif horizontal_actions == 0:
        print("❌ PROBLEM: No horizontal movement detected!")
        print("   The model is not learning to move left/right")
    elif vertical_actions < horizontal_actions * 0.1:
        print("⚠️  WARNING: Very little vertical movement")
        print("   The model heavily prefers horizontal movement")
    elif horizontal_actions < vertical_actions * 0.1:
        print("⚠️  WARNING: Very little horizontal movement")
        print("   The model heavily prefers vertical movement")
    else:
        print("✅ Good: Model uses both vertical and horizontal movement")

def main():
    print("🔍 MODEL ACTION DEBUGGER")
    print("=" * 50)
    
    # Find available models
    model_candidates = [
        "improved_models/quick_model_final.zip",
        "improved_models/quick_enhanced_model_final.zip",
        "improved_models/improved_air_hockey_final.zip",
        "models/air_hockey_ppo_final.zip",
    ]
    
    available_models = []
    for model_path in model_candidates:
        if os.path.exists(model_path):
            available_models.append(model_path)
    
    if not available_models:
        print("No models found to debug")
        return
    
    print("Available models:")
    for i, model_path in enumerate(available_models):
        print(f"{i+1}. {model_path}")
    
    try:
        choice = input(f"\nSelect model to debug (1-{len(available_models)}): ").strip()
        model_idx = int(choice) - 1
        
        if 0 <= model_idx < len(available_models):
            selected_model = available_models[model_idx]
            
            steps = input("Number of steps to test (default: 1000): ").strip()
            steps = int(steps) if steps else 1000
            
            debug_model_actions(selected_model, steps)
        else:
            print("Invalid selection")
    
    except ValueError:
        print("Invalid input")
    except KeyboardInterrupt:
        print("\nDebug interrupted")

if __name__ == "__main__":
    main() 