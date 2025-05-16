# main_with_gym.py
import pygame
import sys
import os
import numpy as np
import torch
from stable_baselines3 import DQN
import time

from constants import *
from sprites import Puck, HumanMallet, AIMallet
from table import Table
from utils import draw_glow
from air_hockey_env import AirHockeyEnv

def load_optimized_model(model_path):
    # Load the model
    model = DQN.load(model_path)
    
    # Set evaluation mode to optimize inference
    model.policy.set_training_mode(False)
    
    # Optimize based on available hardware
    if torch.cuda.is_available():
        model.policy = model.policy.to("cuda")
    else:
        torch.set_num_threads(4)
    
    return model

def main(use_rl=False):
    # Initialize pygame
    pygame.init()
    
    # Create screen and clock
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Air Hockey - Gym RL Integration")
    clock = pygame.time.Clock()
    
    # Create game objects
    table = Table()
    human_mallet = HumanMallet()
    puck = Puck()
    ai_mallet = AIMallet()
    
    # Load RL model if needed
    model = None
    if use_rl:
        try:
            print("Loading RL model...")
            model = load_optimized_model("air_hockey_dqn")
            print("Model loaded successfully!")
            
            # Pre-warm the model with a sample prediction
            # This helps avoid first-frame lag
            dummy_obs = np.zeros((1, 6), dtype=np.float32)
            model.predict(dummy_obs, deterministic=True)
        except Exception as e:
            print(f"Error loading model: {e}")
            use_rl = False
            print("Falling back to simple AI")
    
    # Sprite group
    all_sprites = pygame.sprite.Group()
    all_sprites.add(human_mallet, ai_mallet, puck)
    
    # Game state
    player_score = 0
    ai_score = 0
    
    # RL prediction management
    last_action = 4  # Default to "stay" action
    last_prediction_time = 0
    prediction_interval = 30  # ms between predictions (decreased for better responsiveness)
    
    # Last update time for physics
    last_physics_update = time.time()
    fixed_physics_step = 1/120  # Fixed physics update rate (120Hz)
    
    # For reset message
    show_reset_message = False
    reset_message_timer = 0
    
    # Main game loop
    running = True
    show_fps = False
    
    while running:
        # Start measuring frame time
        frame_start = time.time()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    show_fps = not show_fps
                elif event.key == pygame.K_r:
                    # Complete reset
                    puck.reset(zero_velocity=True)
                    
                    # Reset mallet positions
                    human_mallet.position = [WIDTH // 4, HEIGHT // 2]
                    human_mallet.rect.center = human_mallet.position
                    ai_mallet.position = [WIDTH * 3 // 4, HEIGHT // 2]
                    ai_mallet.rect.center = ai_mallet.position
                    
                    # Reset velocities
                    human_mallet.velocity = [0, 0]
                    ai_mallet.velocity = [0, 0]
                    
                    # Show reset message
                    show_reset_message = True
                    reset_message_timer = pygame.time.get_ticks()
        
        # Direct mouse polling for responsive human mallet
        mouse_pos = pygame.mouse.get_pos()
        human_mallet.update(mouse_pos)
        
        # AI mallet control logic
        if use_rl and model is not None:
            current_time = pygame.time.get_ticks()
            
            # Make predictions at specified intervals
            if current_time - last_prediction_time > prediction_interval:
                # Prepare observation
                observation = np.array([
                    ai_mallet.position[0] / WIDTH,
                    ai_mallet.position[1] / HEIGHT,
                    puck.position[0] / WIDTH,
                    puck.position[1] / HEIGHT,
                    np.clip(puck.velocity[0] / puck.max_speed, -1, 1),
                    np.clip(puck.velocity[1] / puck.max_speed, -1, 1)
                ], dtype=np.float32)
                
                # Run prediction in a way that doesn't block the game loop too much
                try:
                    action, _states = model.predict(observation, deterministic=True)
                    last_action = action
                except Exception as e:
                    print(f"Prediction error: {e}")
                
                last_prediction_time = current_time
            
            # Apply the action with fixed step size
            prev_position = ai_mallet.position.copy()
            move_amount = 5  # Fixed step size
            
            if last_action == 0:  # Up
                ai_mallet.position[1] = max(ai_mallet.position[1] - move_amount, ai_mallet.radius)
            elif last_action == 1:  # Down
                ai_mallet.position[1] = min(ai_mallet.position[1] + move_amount, HEIGHT - ai_mallet.radius)
            elif last_action == 2:  # Left
                ai_mallet.position[0] = max(ai_mallet.position[0] - move_amount, WIDTH // 2 + ai_mallet.radius)
            elif last_action == 3:  # Right
                ai_mallet.position[0] = min(ai_mallet.position[0] + move_amount, WIDTH - ai_mallet.radius)
            
            # Update rect and calculate velocity
            ai_mallet.rect.center = ai_mallet.position
            ai_mallet.velocity = [
                (ai_mallet.position[0] - prev_position[0]),
                (ai_mallet.position[1] - prev_position[1])
            ]
        else:
            # Simple AI behavior
            ai_mallet.update(puck.position)
        
        # Fixed physics time stepping - CRITICAL for consistent puck physics
        current_time = time.time()
        elapsed = current_time - last_physics_update
        
        # Ensure physics runs at a consistent rate regardless of frame rate
        physics_steps = int(elapsed / fixed_physics_step)
        if physics_steps > 0:
            # Run multiple smaller physics steps if needed
            for _ in range(min(physics_steps, 3)):  # Cap at 3 to prevent spiral of death
                # Update puck with fixed time step
                puck.update()
                last_physics_update = current_time
        
        # Check collisions AFTER updating physics
        if puck.check_mallet_collision(human_mallet):
            # Add a small "kick" to ensure the puck doesn't get stuck
            if vector_length(puck.velocity) < 2:
                direction = normalize_vector([
                    puck.position[0] - human_mallet.position[0],
                    puck.position[1] - human_mallet.position[1]
                ])
                puck.velocity[0] += direction[0] * 1
                puck.velocity[1] += direction[1] * 1
        
        if puck.check_mallet_collision(ai_mallet):
            # Same kick for AI collisions
            if vector_length(puck.velocity) < 2:
                direction = normalize_vector([
                    puck.position[0] - ai_mallet.position[0],
                    puck.position[1] - ai_mallet.position[1]
                ])
                puck.velocity[0] += direction[0] * 1
                puck.velocity[1] += direction[1] * 1
        
        # Check for goals
        goal = table.is_goal(puck)
        if goal == "player":
            player_score += 1
            puck.reset("player")
        elif goal == "ai":
            ai_score += 1
            puck.reset("ai")
        
        # Draw everything
        table.draw(screen)
        
        # Draw glows
        draw_glow(screen, (255, 0, 0), human_mallet.position, human_mallet.radius)
        draw_glow(screen, (0, 255, 0), ai_mallet.position, ai_mallet.radius)
        draw_glow(screen, (0, 0, 255), puck.position, puck.radius)
        
        # Draw velocity vector for debugging (optional)
        if show_fps:  # Only show when FPS is enabled
            # Draw puck velocity vector
            end_pos = [
                puck.position[0] + puck.velocity[0] * 5,
                puck.position[1] + puck.velocity[1] * 5
            ]
            pygame.draw.line(screen, (255, 255, 0), puck.position, end_pos, 2)
        
        # Draw sprites
        all_sprites.draw(screen)
        
        # Draw UI elements
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"{player_score} - {ai_score}", True, WHITE)
        screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))
        
        mode_text = font.render("Mode: " + ("Gymnasium RL" if use_rl else "Simple AI"), True, WHITE)
        screen.blit(mode_text, (WIDTH // 2 - mode_text.get_width() // 2, HEIGHT - 30))
        
        controls_text = font.render("F: FPS | R: Reset", True, WHITE)
        screen.blit(controls_text, (10, HEIGHT - 30))
        
        # Show reset message if active
        if show_reset_message:
            if pygame.time.get_ticks() - reset_message_timer < 1500:
                reset_text = font.render("¡Juego Reiniciado!", True, WHITE)
                text_rect = reset_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
                screen.blit(reset_text, text_rect)
            else:
                show_reset_message = False
        
        # Show FPS and debug info
        if show_fps:
            fps = clock.get_fps()
            fps_text = font.render(f"FPS: {int(fps)}", True, WHITE)
            screen.blit(fps_text, (10, 10))
            
            # Show puck velocity
            vel_magnitude = vector_length(puck.velocity)
            vel_text = font.render(f"Puck Vel: {vel_magnitude:.2f}", True, WHITE)
            screen.blit(vel_text, (10, 40))
        
        # Update display
        pygame.display.flip()
        
        # Maintain consistent frame rate
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

# Import utility functions to avoid NameError
from utils import vector_length, normalize_vector

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