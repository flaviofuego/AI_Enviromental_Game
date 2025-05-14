# main_with_gym.py
import pygame
import sys
import os
import numpy as np
from stable_baselines3 import DQN

from constants import *
from sprites import Puck, HumanMallet, AIMallet
from table import Table
from utils import draw_glow
from air_hockey_env import AirHockeyEnv

# At the top of your file
import torch

# When loading the model
def load_optimized_model(model_path):
    # Load the model
    model = DQN.load(model_path)
    
    # Set evaluation mode (disables dropout layers)
    model.policy.set_training_mode(False)
    
    # If possible, use CUDA for faster inference
    if torch.cuda.is_available():
        model.policy = model.policy.to("cuda")
    else:
        # Use CPU with optimizations if CUDA not available
        torch.set_num_threads(4)  # Adjust based on your CPU
    
    return model

# Then in your main function:

    

def main(use_rl=False):
    # Initialize pygame and create objects as before
    
     # Inicializar pygame
    pygame.init()
    
    # Crear pantalla
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Air Hockey - Gym RL Integration")
    clock = pygame.time.Clock()
    
    # Crear objetos del juego
    table = Table()
    human_mallet = HumanMallet()
    ai_mallet = AIMallet()  # O RLAIMallet si prefieres
    puck = Puck()
    
    # If using RL, load the model earlier
    if use_rl:
       model = load_optimized_model("air_hockey_dqn")

    
    # Variables for RL prediction timing
    last_prediction_time = 0
    prediction_interval = 50  # ms between predictions (adjust as needed)
    last_action = 4  # Default to "stay" action
    
    
     # Sprites
    all_sprites = pygame.sprite.Group()
    all_sprites.add(human_mallet, ai_mallet, puck)
    
    # Main game loop
    running = True
    while running:
        # Start timing the frame
        frame_start_time = pygame.time.get_ticks()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Other event handling as before
        
        # Update human mallet FIRST and make it responsive
        mouse_pos = pygame.mouse.get_pos()
        human_mallet.update(mouse_pos)
        
        # Update AI mallet with throttled prediction
        
        # Variables for frame skipping
        frame_skip = 3  # Process RL prediction every 3 frames
        frame_count = 0

        # In main game loop:
        frame_count = (frame_count + 1) % frame_skip
        if use_rl and frame_count == 0:
            current_time = pygame.time.get_ticks()
            
            # Only make predictions at specified intervals
            if current_time - last_prediction_time > prediction_interval:
                # Get observation
                observation = np.array([
                    ai_mallet.position[0] / WIDTH,
                    ai_mallet.position[1] / HEIGHT,
                    puck.position[0] / WIDTH,
                    puck.position[1] / HEIGHT,
                    np.clip(puck.velocity[0] / puck.max_speed, -1, 1),
                    np.clip(puck.velocity[1] / puck.max_speed, -1, 1)
                ], dtype=np.float32)
                
                # Get action from model (with error handling)
                try:
                    action, _states = model.predict(observation, deterministic=True)
                    last_action = action
                except Exception as e:
                    print(f"Prediction error: {e}")
                    # Keep using the last successful action
                
                last_prediction_time = current_time
            
            # Apply the most recent action
            prev_position = ai_mallet.position.copy()
            move_amount = 5
            
            if last_action == 0:  # Up
                ai_mallet.position[1] = max(ai_mallet.position[1] - move_amount, ai_mallet.radius)
            elif last_action == 1:  # Down
                ai_mallet.position[1] = min(ai_mallet.position[1] + move_amount, HEIGHT - ai_mallet.radius)
            elif last_action == 2:  # Left
                ai_mallet.position[0] = max(ai_mallet.position[0] - move_amount, WIDTH // 2 + ai_mallet.radius)
            elif last_action == 3:  # Right
                ai_mallet.position[0] = min(ai_mallet.position[0] + move_amount, WIDTH - ai_mallet.radius)
            # Action 4 = stay
            
            # Update rect and calculate velocity
            ai_mallet.rect.center = ai_mallet.position
            ai_mallet.velocity = [
                ai_mallet.position[0] - prev_position[0],
                ai_mallet.position[1] - prev_position[1]
            ]
        else:
            # Use simple AI behavior
            ai_mallet.update(puck.position)
        
        # Rest of game logic (update puck, check collisions, etc.)
        
        # Update display
        pygame.display.flip()
        
        # Maintain consistent framerate
        elapsed = pygame.time.get_ticks() - frame_start_time
        if elapsed < (1000 / FPS):
            pygame.time.delay(int((1000 / FPS) - elapsed))


if __name__ == "__main__":
    print("=== Air Hockey with Gymnasium Reinforcement Learning ===")
    print("\nOptions:")
    print("1. Play against Simple AI")
    print("2. Play against Gymnasium-trained RL agent")
    
    choice = input("\nSelect an option (1-2): ").strip()
    
    if choice == "1":
        main(use_rl=False)
    elif choice == "2":
        model_path = "air_hockey_dqn.zip"
        
        if os.path.exists(model_path):
            main(use_rl=True)
        else:
            print(f"Trained model not found at {model_path}")
            choice = input("Do you want to play with Simple AI instead? (y/n): ").lower().startswith('y')
            if choice:
                main(use_rl=False)
            else:
                print("Please run train_agent.py first to train a model")
    else:
        print("Invalid option, starting with Simple AI")
        main(use_rl=False)