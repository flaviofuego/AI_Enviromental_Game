# main_improved.py
import pygame
import sys
import os
import numpy as np
import torch
import time

torch.cuda.is_available = lambda: False

# Performance optimization settings
os.environ['OMP_NUM_THREADS'] = '4'
torch.set_num_threads(4)

# Try importing PPO first (for newer models), fall back to DQN if not available
try:
    from stable_baselines3 import PPO
    ModelClass = PPO
except ImportError:
    from stable_baselines3 import DQN
    ModelClass = DQN

from constants import *
from sprites import Puck, HumanMallet, AIMallet
from table import Table
from utils import draw_glow, vector_length, normalize_vector

def load_optimized_model(model_path, model_type="auto"):
    """Load and optimize the RL model for maximum performance"""
    print(f"Loading model from: {model_path}")
    
    # Try to load with PPO first (most common for our models)
    model = None
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        print("Loaded as PPO model")
    except Exception as e:
        print(f"Failed to load as PPO: {e}")
        try:
            from stable_baselines3 import DQN
            model = DQN.load(model_path)
            print("Loaded as DQN model")
        except Exception as e2:
            print(f"Failed to load as DQN: {e2}")
            raise Exception(f"Could not load model with either PPO or DQN: {e}, {e2}")
    
    # Auto-detect model type based on observation space
    obs_space = model.observation_space
    if hasattr(obs_space, 'shape'):
        obs_dim = obs_space.shape[0]
        if obs_dim == 21:
            model_type = "enhanced"  # New enhanced models
            print(f"Detected enhanced model with {obs_dim} dimensions")
        elif obs_dim == 13:
            model_type = "original"  # Original models
            print(f"Detected original model with {obs_dim} dimensions")
        else:
            model_type = "unknown"
            print(f"Unknown model type with {obs_dim} dimensions")
    else:
        model_type = "unknown"
        print("Could not determine model type")
    
    # Force evaluation mode for faster inference
    model.policy.set_training_mode(False)
    
    # Optimize based on available hardware
    if torch.cuda.is_available():
        model.policy = model.policy.to("cuda")
        print("Using CUDA for model inference")
    else:
        torch.set_num_threads(4)
        print("Using CPU for model inference with 4 threads")
    
    print(f"Model type detected: {model_type}")
    return model, model_type

def create_observation_for_model(ai_mallet, puck, human_mallet, player_score, ai_score, model_type="original"):
    """Create observation vector based on model type"""
    
    if model_type == "enhanced":
        # Enhanced observation for new enhanced models (21 dimensions)
        # This matches the EnhancedAirHockeyEnv observation space
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
        time_factor = 0.5  # Placeholder
        score_diff = (ai_score - player_score) / 7.0
        
        # Predicción de trayectoria del puck
        if abs(puck.velocity[0]) > 0.1:
            time_to_ai_side = (WIDTH - puck.position[0]) / puck.velocity[0] if puck.velocity[0] > 0 else 0
            predicted_y_at_ai_side = puck.position[1] + puck.velocity[1] * time_to_ai_side
            predicted_y_norm = np.clip(predicted_y_at_ai_side / HEIGHT, 0, 1)
        else:
            predicted_y_norm = puck_y_norm
        
        # Construir vector de observación (21 dimensiones) - EXACTAMENTE como en EnhancedAirHockeyEnv
        observation = np.array([
            # Posiciones (6)
            ai_x_norm, ai_y_norm,
            puck_x_norm, puck_y_norm,
            human_x_norm, human_y_norm,
            
            # Velocidades (6)
            puck_vx_norm, puck_vy_norm,
            ai_vx_norm, ai_vy_norm,
            human_vx_norm, human_vy_norm,
            
            # Distancias (2)
            puck_to_ai_dist, puck_to_human_dist,
            
            # Información contextual (7)
            puck_in_ai_half,
            puck_moving_to_ai_goal,
            puck_moving_to_human_goal,
            time_factor,
            score_diff,
            predicted_y_norm,
            0.5  # Nivel de dificultad normalizado (placeholder)
        ], dtype=np.float32)
        
    elif model_type == "improved":
        # Legacy improved observation for older improved models (21 dimensions)
        # Normalizar posiciones
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
        time_factor = 0.5  # Placeholder
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
            # Posiciones (6)
            ai_x_norm, ai_y_norm,
            puck_x_norm, puck_y_norm,
            human_x_norm, human_y_norm,
            
            # Velocidades (6)
            puck_vx_norm, puck_vy_norm,
            ai_vx_norm, ai_vy_norm,
            human_vx_norm, human_vy_norm,
            
            # Distancias (2)
            puck_to_ai_dist, puck_to_human_dist,
            
            # Información contextual (7)
            puck_in_ai_half,
            puck_moving_to_ai_goal,
            puck_moving_to_human_goal,
            time_factor,
            score_diff,
            predicted_y_norm,
            0.5  # Nivel de dificultad normalizado (placeholder)
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
        
        # Calculate additional features for 13-dim models
        puck_to_mallet_dist = np.sqrt(
            (puck.position[0] - ai_mallet.position[0])**2 + 
            (puck.position[1] - ai_mallet.position[1])**2
        ) / np.sqrt(WIDTH**2 + HEIGHT**2)
        
        puck_to_ai_goal = (WIDTH - puck.position[0]) / WIDTH
        puck_to_player_goal = puck.position[0] / WIDTH
        time_since_hit = 0.5
        puck_moving_to_player = 1.0 if puck.velocity[0] < 0 else 0.0
        
        # Create full observation vector
        observation = np.append(basic_obs, [
            puck_to_mallet_dist,
            puck_to_ai_goal,
            puck_to_player_goal,
            time_since_hit,
            puck_moving_to_player,
            player_score / 5.0,
            ai_score / 5.0
        ])
    
    return observation

def find_best_model():
    """Find the best available model automatically"""
    model_candidates = [
        # Fixed models (highest priority - completely corrected behavior)
        ("improved_models/fixed_air_hockey_final.zip", "enhanced"),
        ("improved_models/quick_fixed_model_final.zip", "enhanced"),
        # Enhanced models (high priority - new vertical movement models)
        ("improved_models/quick_enhanced_model_final.zip", "enhanced"),
        ("improved_models/enhanced_vertical_model_final.zip", "enhanced"),
        # Legacy improved models
        ("improved_models/improved_air_hockey_final.zip", "improved"),
        ("improved_models/quick_model_final.zip", "improved"),
        # Original models
        ("models/air_hockey_ppo_final.zip", "original"),
        ("air_hockey_dqn.zip", "original"),
    ]
    
    for model_path, model_type in model_candidates:
        if os.path.exists(model_path):
            return model_path, model_type
    
    return None, None

def main(use_rl=False, model_path=None):
    """Main game function with support for improved models"""
    # Initialize pygame
    pygame.init()
    
    # Create screen and clock
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Air Hockey - Improved RL System")
    clock = pygame.time.Clock()
    
    # Create game objects
    table = Table()
    human_mallet = HumanMallet()
    puck = Puck()
    ai_mallet = AIMallet()
    
    # Load RL model if needed
    model = None
    model_type = "original"
    
    if use_rl:
        try:
            print("Loading RL model...")
            
            # Use specified model or find best available
            if model_path and os.path.exists(model_path):
                model, model_type = load_optimized_model(model_path)
            else:
                auto_model_path, auto_model_type = find_best_model()
                if auto_model_path:
                    model, model_type = load_optimized_model(auto_model_path, auto_model_type)
                    print(f"Auto-selected model: {auto_model_path}")
                else:
                    print("No model file found")
                    use_rl = False
                    
            if model:
                print("Model loaded successfully!")
                print(f"Model type: {model_type}")
                
                # Pre-warm the model with appropriate observation size
                if model_type == "enhanced":
                    dummy_obs = np.zeros((21,), dtype=np.float32)
                elif model_type == "improved":
                    dummy_obs = np.zeros((21,), dtype=np.float32)
                else:
                    dummy_obs = np.zeros((13,), dtype=np.float32)
                model.predict(dummy_obs, deterministic=True)
            else:
                print("Failed to load model")
                use_rl = False
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
    prediction_interval = 20  # ms between predictions
    
    # Behavioral correction system
    force_vertical_threshold = 80  # Force vertical movement if puck is this far vertically
    vertical_move_cooldown = 15  # Frames between forced vertical moves
    last_vertical_move = 100  # Time since last vertical move
    force_horizontal_threshold = 120  # Force horizontal movement if puck is this far horizontally
    horizontal_move_cooldown = 15  # Frames between forced horizontal moves
    last_horizontal_move = 100  # Time since last horizontal move
    movement_history = []  # Track recent actions
    stuck_in_bottom_counter = 0  # Counter for being stuck in bottom
    stuck_in_side_counter = 0  # Counter for being stuck in side
    
    # Frame skip for AI updates
    frame_count = 0
    frame_skip = 2
    
    # Physics time step control
    last_physics_update = time.time()
    fixed_physics_step = 1/120
    
    # For reset message
    show_reset_message = False
    reset_message_timer = 0
    
    # Main game loop
    running = True
    show_fps = False
    
    # Precompute some values
    half_width = WIDTH // 2
    
    while running:
        frame_start = time.time()
        
        # Handle all events at once
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
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    # Switch model type (if multiple available)
                    if use_rl:
                        print("Switching to next available model...")
                        # Find next model
                        current_improved = model_type == "improved"
                        if current_improved:
                            # Try Fixed models
                            for path in ["improved_models/fixed_air_hockey_final.zip", "improved_models/quick_fixed_model_final.zip"]:
                                if os.path.exists(path):
                                    try:
                                        model, model_type = load_optimized_model(path, "enhanced")
                                        print(f"Switched to: {path}")
                                        break
                                    except:
                                        continue
                        else:
                            # Try improved models
                            for path in ["improved_models/improved_air_hockey_final.zip", "improved_models/quick_model_final.zip"]:
                                if os.path.exists(path):
                                    try:
                                        model, model_type = load_optimized_model(path, "improved")
                                        print(f"Switched to: {path}")
                                        break
                                    except:
                                        continue
        
        # Direct mouse polling for responsive human mallet
        mouse_pos = pygame.mouse.get_pos()
        human_mallet.update(mouse_pos)
        
        # AI mallet control logic - with frame skipping
        frame_count = (frame_count + 1) % frame_skip
        
        if use_rl and model is not None and frame_count == 0:
            current_time = pygame.time.get_ticks()
            
            # Make predictions at specified intervals
            if current_time - last_prediction_time > prediction_interval:
                # Create observation based on model type
                observation = create_observation_for_model(
                    ai_mallet, puck, human_mallet, player_score, ai_score, model_type
                )
                
                # Make prediction with error handling
                try:
                    action, _states = model.predict(observation, deterministic=True)
                    # Convert numpy array to int if necessary
                    if isinstance(action, np.ndarray):
                        if action.ndim == 0:  # 0-dimensional array
                            action = int(action.item())
                        else:
                            action = int(action[0])
                    else:
                        action = int(action)

                    
                    # BEHAVIORAL CORRECTION SYSTEM
                    last_vertical_move += 1
                    last_horizontal_move += 1
                    movement_history.append(action)
                    if len(movement_history) > 20:
                        movement_history.pop(0)
                    
                    # Check if AI is stuck in the bottom of the field
                    if ai_mallet.position[1] > HEIGHT * 0.75:
                        stuck_in_bottom_counter += 1
                    else:
                        stuck_in_bottom_counter = 0
                    
                    # Check if AI is stuck on the sides (too close to AI goal or center line)
                    if (ai_mallet.position[0] > WIDTH * 0.85 or  # Too close to AI goal
                        ai_mallet.position[0] < WIDTH * 0.55):   # Too close to center line
                        stuck_in_side_counter += 1
                    else:
                        stuck_in_side_counter = 0
                    
                    # Calculate distances and game state
                    original_action = action
                    y_distance = abs(puck.position[1] - ai_mallet.position[1])
                    x_distance = abs(puck.position[0] - ai_mallet.position[0])
                    puck_in_ai_half = puck.position[0] > WIDTH // 2
                    
                    # Count recent movements
                    recent_vertical = sum(1 for a in movement_history[-10:] if a in [0, 1]) if len(movement_history) >= 10 else 0
                    recent_horizontal = sum(1 for a in movement_history[-10:] if a in [2, 3]) if len(movement_history) >= 10 else 0
                    
                    # Force movement conditions
                    force_vertical = False
                    force_horizontal = False
                    
                    # VERTICAL MOVEMENT FORCING
                    # 1. Puck is far vertically and AI hasn't moved vertically recently
                    if (y_distance > force_vertical_threshold and 
                        last_vertical_move > vertical_move_cooldown and 
                        puck_in_ai_half and recent_vertical < 2):
                        force_vertical = True
                        reason = "puck_far_vertical"
                    
                    # 2. AI is stuck in bottom corner
                    elif stuck_in_bottom_counter > 5 and recent_vertical == 0:
                        force_vertical = True
                        reason = "stuck_in_bottom"
                    
                    # 3. No vertical movement in recent history and puck is in AI half
                    elif (len(movement_history) >= 15 and recent_vertical == 0 and 
                          puck_in_ai_half and y_distance > 40):
                        force_vertical = True
                        reason = "no_recent_vertical"
                    
                    # HORIZONTAL MOVEMENT FORCING
                    # 1. Puck is far horizontally and AI hasn't moved horizontally recently
                    if (not force_vertical and x_distance > force_horizontal_threshold and 
                        last_horizontal_move > horizontal_move_cooldown and 
                        puck_in_ai_half and recent_horizontal < 2):
                        force_horizontal = True
                        reason = "puck_far_horizontal"
                    
                    # 2. AI is stuck on the sides
                    elif (not force_vertical and stuck_in_side_counter > 5 and recent_horizontal == 0):
                        force_horizontal = True
                        reason = "stuck_in_side"
                    
                    # 3. No horizontal movement in recent history and puck is in AI half
                    elif (not force_vertical and len(movement_history) >= 15 and recent_horizontal == 0 and 
                          puck_in_ai_half and x_distance > 60):
                        force_horizontal = True
                        reason = "no_recent_horizontal"
                    
                    # Apply forced movements (vertical has priority)
                    if force_vertical:
                        if puck.position[1] < ai_mallet.position[1]:
                            action = 0  # Up
                        else:
                            action = 1  # Down
                        last_vertical_move = 0
                        
                        # Debug info (optional)
                        if show_fps:
                            print(f"Forced vertical movement: {action} (reason: {reason})")
                    
                    elif force_horizontal:
                        if puck.position[0] > ai_mallet.position[0]:
                            action = 3  # Right
                        else:
                            action = 2  # Left
                        last_horizontal_move = 0
                        
                        # Debug info (optional)
                        if show_fps:
                            print(f"Forced horizontal movement: {action} (reason: {reason})")
                    
                    last_action = action
                except Exception as e:
                    print(f"Prediction error: {e}")
                
                last_prediction_time = current_time
            
            # Apply the action with fixed step size
            prev_position = ai_mallet.position.copy()
            move_amount = 7
            
            if last_action == 0:  # Up
                ai_mallet.position[1] = max(ai_mallet.position[1] - move_amount, ai_mallet.radius)
            elif last_action == 1:  # Down
                ai_mallet.position[1] = min(ai_mallet.position[1] + move_amount, HEIGHT - ai_mallet.radius)
            elif last_action == 2:  # Left
                ai_mallet.position[0] = max(ai_mallet.position[0] - move_amount, half_width + ai_mallet.radius)
            elif last_action == 3:  # Right
                ai_mallet.position[0] = min(ai_mallet.position[0] + move_amount, WIDTH - ai_mallet.radius)
            
            # Update rect and calculate velocity
            ai_mallet.rect.center = ai_mallet.position
            ai_mallet.velocity = [
                (ai_mallet.position[0] - prev_position[0]),
                (ai_mallet.position[1] - prev_position[1])
            ]
        else:
            # Use simple AI behavior when RL is not active
            if not use_rl:
                ai_mallet.update(puck.position)
        
        # Fixed physics time stepping
        current_time = time.time()
        elapsed = current_time - last_physics_update
        
        if elapsed >= fixed_physics_step:
            puck.update()
            last_physics_update = current_time
        
        # Check collisions
        if puck.check_mallet_collision(human_mallet):
            if vector_length(puck.velocity) < 2:
                direction = normalize_vector([
                    puck.position[0] - human_mallet.position[0],
                    puck.position[1] - human_mallet.position[1]
                ])
                puck.velocity[0] += direction[0] * 1
                puck.velocity[1] += direction[1] * 1
        
        if puck.check_mallet_collision(ai_mallet):
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
        current_fps = clock.get_fps()
        if not show_fps or current_fps == 0 or current_fps > 40:
            draw_glow(screen, (255, 0, 0), human_mallet.position, human_mallet.radius)
            draw_glow(screen, (0, 255, 0), ai_mallet.position, ai_mallet.radius)
            draw_glow(screen, (0, 0, 255), puck.position, puck.radius)
        
        # Draw velocity vector for debugging
        if show_fps:
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
        
        # Show model type and mode
        if use_rl:
            if "fixed" in str(model).lower() if model else False:
                mode_text = font.render("Mode: Fixed RL (Full Movement Corrections)", True, WHITE)
            elif model_type == "enhanced":
                mode_text = font.render("Mode: Enhanced RL (Vertical & Horizontal Movement)", True, WHITE)
            else:
                mode_text = font.render(f"Mode: {model_type.title()} RL (Movement Corrections)", True, WHITE)
        else:
            mode_text = font.render("Mode: Simple AI", True, WHITE)
        screen.blit(mode_text, (WIDTH // 2 - mode_text.get_width() // 2, HEIGHT - 50))
        
        controls_text = font.render("F: FPS | R: Reset | M: Switch Model | ESC: Quit", True, WHITE)
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
            
            # Show AI info if using RL
            if use_rl and model:
                action_names = ["Up", "Down", "Left", "Right", "Stay"]
                action_text = font.render(f"AI Action: {action_names[last_action]}", True, WHITE)
                screen.blit(action_text, (10, 70))
                
                # Show model info
                model_info = font.render(f"Model: {model_type}", True, WHITE)
                screen.blit(model_info, (10, 100))
        
        # Update display
        pygame.display.flip()
        
        # Maintain consistent frame rate
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    print("=== Air Hockey with Improved RL System ===")
    print("\nOptions:")
    print("1. Play against Simple AI")
    print("2. Play against RL agent (auto-detect best model)")
    print("3. Play against specific model")
    
    try:
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == "1":
            main(use_rl=False)
        elif choice == "2":
            # Auto-detect best model
            model_path, model_type = find_best_model()
            if model_path:
                print(f"Found model: {model_path} (type: {model_type})")
                main(use_rl=True)
            else:
                print("No trained model found")
                fallback = input("Do you want to play with Simple AI instead? (y/n): ").lower().startswith("y")
                if fallback:
                    main(use_rl=False)
                else:
                    print("Please train a model first")
        elif choice == "3":
                    print("\nAvailable models:")
        models = []
        
        fixed_models = [
            "improved_models/fixed_air_hockey_final.zip",
            "improved_models/quick_fixed_model_final.zip"
        ]
        for model_path in fixed_models:
            if os.path.exists(model_path):
                models.append((model_path, "fixed"))
                print(f"{len(models)}. {model_path} (Fixed)")
                
        
        # Check for enhanced models (new vertical movement models)
        enhanced_models = [
            "improved_models/quick_enhanced_model_final.zip",
            "improved_models/enhanced_vertical_model_final.zip"
        ]
        for model_path in enhanced_models:
            if os.path.exists(model_path):
                models.append((model_path, "enhanced"))
                print(f"{len(models)}. {model_path} (Enhanced - Vertical Movement)")
        
        # Check for legacy improved models
        improved_models = [
            "improved_models/improved_air_hockey_final.zip",
            "improved_models/quick_model_final.zip"
        ]
        for model_path in improved_models:
            if os.path.exists(model_path):
                models.append((model_path, "improved"))
                print(f"{len(models)}. {model_path} (Legacy Improved)")
        
        # Check for original models
        original_models = [
            "models/air_hockey_ppo_final.zip",
            "air_hockey_dqn.zip"
        ]
        for model_path in original_models:
            if os.path.exists(model_path):
                models.append((model_path, "original"))
                print(f"{len(models)}. {model_path} (Original)")
            
            if not models:
                print("No models found. Please train a model first.")
            else:
                try:
                    model_choice = int(input(f"\nSelect model (1-{len(models)}): ")) - 1
                    if 0 <= model_choice < len(models):
                        selected_model, model_type = models[model_choice]
                        print(f"Using: {selected_model}")
                        main(use_rl=True, model_path=selected_model)
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Invalid input")
        else:
            print("Invalid option, starting with Simple AI")
            main(use_rl=False)
    except KeyboardInterrupt:
        print("\nExiting the game.")
        pygame.quit()
        sys.exit() 