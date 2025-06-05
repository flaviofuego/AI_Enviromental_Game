# main_improved.py
import pygame
import sys
import os
import numpy as np
import torch
import time


# Performance optimization settings
torch.cuda.is_available = lambda: False
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

def main(level_id=None, debug_mode=True, use_rl=True):
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
    
    try:
        print("Loading RL model...")
        
        # Auto-detect best model
        model_path, model_type = find_best_model()
        if model_path:
            model, model_type = load_optimized_model(model_path)
            print(f"Auto-selected model: {model_path}")
            print("Model loaded successfully!")
            print(f"Model type: {model_type}")
            
            # Pre-warm the model
            if model_type == "enhanced":
                dummy_obs = np.zeros((21,), dtype=np.float32)
            elif model_type == "improved":
                dummy_obs = np.zeros((21,), dtype=np.float32)
            else:
                dummy_obs = np.zeros((13,), dtype=np.float32)
            model.predict(dummy_obs, deterministic=True)
        else:
            print("No model file found")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Sprite group
    all_sprites = pygame.sprite.Group()
    all_sprites.add(human_mallet, ai_mallet, puck)
    
    # Game state
    player_score = 0
    ai_score = 0
    
    # RL prediction management
    last_action = 4  # Default to "stay" action
    last_prediction_time = 0
    prediction_interval = 15  # ms between predictions (reducido de 20 para más responsividad)
    
    # Behavioral correction system
    force_vertical_threshold = 80
    vertical_move_cooldown = 15
    last_vertical_move = 100
    force_horizontal_threshold = 120
    horizontal_move_cooldown = 15
    last_horizontal_move = 100
    movement_history = []
    stuck_in_bottom_counter = 0
    stuck_in_side_counter = 0
    
    # Frame skip for AI updates
    frame_count = 0
    frame_skip = 1  # Reducido de 2 para actualizaciones más frecuentes
    
    # Physics time step control
    last_physics_update = time.time()
    fixed_physics_step = 1/120  # Ajustado para 120 FPS
    
    # For reset message
    show_reset_message = False
    reset_message_timer = 0
    
    # Game over state
    game_over = False
    winner = None
    
    # Button dimensions for retry
    button_width = 200
    button_height = 50
    button_x = WIDTH // 2 - button_width // 2
    button_y = HEIGHT // 2 + 50
    
    # Main game loop
    running = True
    show_fps = False
    
    # Get level configuration if level_id is provided
    if level_id is not None:
        from game.config.level_config import get_level_config
        level_config = get_level_config(level_id)
        table.table_color = level_config["theme"]["table_color"]
    
    while running:
        frame_start = time.time()
        
        # Get mouse position for both game and UI interaction
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        # Handle all events at once
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if debug_mode:
                    if event.key == pygame.K_f:
                        show_fps = not show_fps
                    elif event.key == pygame.K_r and not game_over:
                        # Complete reset
                        puck.reset(zero_velocity=True)
                        human_mallet.position = [WIDTH // 4, HEIGHT // 2]
                        human_mallet.rect.center = human_mallet.position
                        ai_mallet.position = [WIDTH * 3 // 4, HEIGHT // 2]
                        ai_mallet.rect.center = ai_mallet.position
                        human_mallet.velocity = [0, 0]
                        ai_mallet.velocity = [0, 0]
                        show_reset_message = True
                        reset_message_timer = pygame.time.get_ticks()
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Only update game state if not in game over
        if not game_over:
            # Update human mallet with the mouse position we got earlier
            human_mallet.update(mouse_pos)
            
            # AI mallet control logic - with frame skipping
            frame_count = (frame_count + 1) % frame_skip
            
            if model is not None and frame_count == 0:
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
                move_amount = 9  # Aumentado de 7 para movimiento más rápido
                
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
                # Use simple AI behavior when RL is not active
                if not use_rl:
                    ai_mallet.update(puck.position, puck.velocity)

            
            # Fixed physics time stepping - optimizado para 120 FPS
            current_time = time.time()
            elapsed = current_time - last_physics_update
            
            # Actualizar física con timestep fijo
            accumulator = elapsed
            while accumulator >= fixed_physics_step:
                puck.update()
                accumulator -= fixed_physics_step
            
            if accumulator < elapsed:
                last_physics_update = current_time - accumulator
            
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
            
            # Check for goal collision (rebote en estructuras de porterías)
            table.check_goal_collision(puck)
            
            # Check for goals
            goal = table.is_goal(puck)
            if goal == "player":
                player_score += 1
                puck.reset("player")
                if player_score >= 7:
                    game_over = True
                    winner = "player"
                    # Stop puck and mallets when game is over
                    puck.velocity = [0, 0]
                    human_mallet.velocity = [0, 0]
                    ai_mallet.velocity = [0, 0]
            elif goal == "ai":
                ai_score += 1
                puck.reset("ai")
                if ai_score >= 7:
                    game_over = True
                    winner = "ai"
                    # Stop puck and mallets when game is over
                    puck.velocity = [0, 0]
                    human_mallet.velocity = [0, 0]
                    ai_mallet.velocity = [0, 0]
        
        # Draw everything
        # Dibujar fondo personalizado si está disponible
        has_custom_background = 'background' in custom_sprites and custom_sprites['background'] is not None
        if has_custom_background:
            screen.blit(custom_sprites['background'], (0, 0))
            # No dibujar fondo sólido en la mesa porque ya tenemos fondo personalizado
            table.draw(screen, draw_background=False, debug_mode=show_fps)
        else:
            # Si no hay fondo personalizado, dejar que la mesa dibuje su fondo
            table.draw(screen, draw_background=True, debug_mode=show_fps)
        
        # Draw glows - optimizado para mejor rendimiento
        current_fps = clock.get_fps()
        # Solo dibujar brillos si FPS es bueno o si no estamos mostrando FPS
        if not show_fps or current_fps > 50:
            # Dibujar brillos completos
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

        # Draw game over screen if game is over
        if game_over:
            # Semi-transparent overlay
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (0, 0))
            
            # Game over text
            game_over_text = "¡Has Ganado!" if winner == "player" else "¡Has Perdido!"
            game_over_surface = font.render(game_over_text, True, WHITE)
            game_over_rect = game_over_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
            screen.blit(game_over_surface, game_over_rect)
            
            # Draw retry button
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
            button_color = (70, 70, 70) if button_rect.collidepoint(mouse_pos) else (50, 50, 50)
            pygame.draw.rect(screen, button_color, button_rect)
            pygame.draw.rect(screen, WHITE, button_rect, 2)
            retry_text = font.render("Reintentar", True, WHITE)
            retry_rect = retry_text.get_rect(center=(WIDTH // 2, button_y + button_height // 2))
            screen.blit(retry_text, retry_rect)
            
            # Check for button click
            if button_rect.collidepoint(mouse_pos) and pygame.mouse.get_pressed()[0]:
                # Reset game
                player_score = 0
                ai_score = 0
                game_over = False
                winner = None
                puck.reset(zero_velocity=True)
                human_mallet.position = [WIDTH // 4, HEIGHT // 2]
                human_mallet.rect.center = human_mallet.position
                ai_mallet.position = [WIDTH * 3 // 4, HEIGHT // 2]
                ai_mallet.rect.center = ai_mallet.position
                human_mallet.velocity = [0, 0]
                ai_mallet.velocity = [0, 0]
                show_reset_message = True
                reset_message_timer = pygame.time.get_ticks()
        
        # Show model type and mode
        if "fixed" in str(model).lower() if model else False:
            mode_text = font.render("Mode: Fixed RL (Full Movement Corrections)", True, WHITE)
        elif model_type == "enhanced":
            mode_text = font.render("Mode: Enhanced RL (Vertical & Horizontal Movement)", True, WHITE)
        else:
            mode_text = font.render(f"Mode: {model_type.title()} RL (Movement Corrections)", True, WHITE)
        screen.blit(mode_text, (WIDTH // 2 - mode_text.get_width() // 2, HEIGHT - 50))
        
        controls_text = font.render("F: FPS | R: Reset | ESC: Quit", True, WHITE)
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
            if model:
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
            "improved_models/quick_fixed_model_final.zip",
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

def start_game_with_level(level_id, save_system=None, screen=None):
    """
    Inicia el juego con configuración específica del nivel
    
    Args:
        level_id (int): ID del nivel seleccionado (1-5)
        save_system: Sistema de guardado del juego principal
        screen: Pantalla de pygame existente (opcional)
    
    Returns:
        dict: Resultados del juego (victoria, puntaje, etc.)
    """
    
    # Configuración específica por nivel
    level_configs = {
        1: {
            'name': "Basura en el Ártico",
            'enemy': "SLICKWAVE",
            'theme': "Plástico",
            'difficulty': 'easy',
            'model_preference': 'fixed',#cambiar a improved
            'background_color': (173, 216, 230)  # Azul hielo
        },
        2: {
            'name': "Agujero de Ozono",
            'enemy': "UVBLADE", 
            'theme': "Gases CFC",
            'difficulty': 'medium',
            'model_preference': 'enhanced',
            'background_color': (255, 140, 0)  # Naranja advertencia
        },
        3: {
            'name': "Tormenta de Smog",
            'enemy': "SMOGATRON",
            'theme': "Aire Contaminado", 
            'difficulty': 'medium',
            'model_preference': 'enhanced',
            'background_color': (100, 100, 150)  # Gris azulado
        },
        4: {
            'name': "Bosque Desvanecido",
            'enemy': "DEFORESTIX",
            'theme': "Deforestación",
            'difficulty': 'hard',
            'model_preference': 'fixed',
            'background_color': (34, 139, 34)  # Verde esperanza
        },
        5: {
            'name': "Isla de Calor Urbano",
            'enemy': "HEATCORE",
            'theme': "Calentamiento Urbano",
            'difficulty': 'very_hard', 
            'model_preference': 'fixed',
            'background_color': (220, 50, 50)  # Rojo crítico
        }
    }
    
    if level_id not in level_configs:
        raise ValueError(f"Nivel {level_id} no válido")
    
    config = level_configs[level_id]
    
    # Determinar modelo RL a usar basado en la dificultad del nivel
    model_path = None
    if config['model_preference'] == 'fixed':
        # Niveles difíciles usan modelos con correcciones
        for path in ["improved_models/quick_fixed_model_final.zip","improved_models/fixed_air_hockey_final.zip" ]:
            if os.path.exists(path):
                model_path = path
                print(f"Using model: {model_path}") 
                break
    elif config['model_preference'] == 'enhanced':
        # Niveles medios usan modelos mejorados
        for path in ["improved_models/quick_enhanced_model_final.zip", "improved_models/enhanced_vertical_model_final.zip"]:
            if os.path.exists(path):
                model_path = path
                print(f"Using model: {model_path}") 
                
                break
    else:  # improved
        # Niveles fáciles usan modelos legacy mejorados
        for path in ["improved_models/quick_model_final.zip"]:
            if os.path.exists(path):
                model_path = path
                print(f"Using model: {model_path}") 
                break
    
    # Configurar ventana si no se proporciona
    if screen is None:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f"Hockey Is Melting Down - {config['name']}")
        own_screen = True
    else:
        own_screen = False
    
    try:
        # Llamar a la función main con configuración específica
        result = main_with_config(
            use_rl=model_path is not None,
            model_path=model_path,
            screen=screen,
            level_config=config,
            save_system=save_system
        )
        
        # Actualizar progreso en el sistema de guardado
        if save_system and result.get('victory'):
            # Crear datos del nivel completado
            level_data = {
                "points": result.get('player_score', 0) * 100,  # Puntos basados en el puntaje
                "level_completed": level_id,
                "enemy_defeated": config['enemy'],
                "planetary_progress": {
                    # Progreso específico según el tema del nivel
                    "oceanos_limpiados": 1 if level_id == 1 else 0,
                    "ozono_restaurado": 1 if level_id == 2 else 0,
                    "aire_purificado": 1 if level_id == 3 else 0,
                    "bosques_replantados": 1 if level_id == 4 else 0,
                    "ciudades_enfriadas": 1 if level_id == 5 else 0,
                },
                "stats": {
                    "games_played": 1,
                    "wins": 1 if result.get('victory') else 0,
                    "losses": 0 if result.get('victory') else 1,
                    "time_played": 300  # Aprox 5 minutos por partida
                }
            }
            save_system.update_game_progress(level_data)
        
        return result
        
    finally:
        if own_screen:
            pygame.quit()

def draw_round_button(screen, color, center, radius, text, font_size=36):
    """Dibuja un botón circular con texto centrado"""
    pygame.draw.circle(screen, color, center, radius)
    pygame.draw.circle(screen, WHITE, center, radius, 2)  # Borde blanco
    
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect(center=center)
    screen.blit(text_surface, text_rect)
    return pygame.Rect(center[0]-radius, center[1]-radius, radius*2, radius*2)

from game.config.save_system import GameSaveSystem


def main_with_config(use_rl=False, model_path=None, screen=None, level_config=None, save_system=None):
    """
    Función principal del juego con configuración específica de nivel
    
    Args:
        use_rl: Si usar modelo de RL
        model_path: Ruta del modelo a cargar
        screen: Pantalla de pygame existente
        level_config: Configuración del nivel
        save_system: Sistema de guardado
    
    Returns:
        dict con resultados del juego
    """
    save_system = GameSaveSystem()

    profiles = save_system.get_all_profiles()
    if profiles:
        last_profile_id = profiles[0]['profile_id']
        profile = save_system.load_profile(last_profile_id)
    else:
        profile = None

    selected_skin_id = profile.get("skin", "default") if profile else "default"

    skin_colors = {
        'default': (200, 200, 200),
        'eco_warrior': (100, 200, 100),
        'arctic': (150, 220, 255),
        'volcano': (255, 100, 50),
        'cyber': (100, 255, 200),
        'retro': (255, 200, 100),
        'scientist': (200, 200, 255),
        'agent': (50, 50, 100),
    }

    mallet_color = skin_colors.get(selected_skin_id, (200, 200, 200))
    
    if screen is None:
        pygame.init()
        info = pygame.display.Info()
        screen_width = min(1200, info.current_w - 100)
        screen_height = min(800, info.current_h - 100)
        screen = pygame.display.set_mode((screen_width, screen_height))
    else:
        screen_width, screen_height = screen.get_size()
    
    # Establecer dimensiones actuales
    set_screen_dimensions(screen_width, screen_height)
    current_width = screen_width
    current_height = screen_height
    
    clock = pygame.time.Clock()
    
    # Cargar sprites personalizados si hay configuración de nivel
    custom_sprites = {}
    if level_config and 'level_id' in level_config:
        from sprite_loader import SpriteLoader
        from game.config.level_config import get_level_config
        
        # Obtener configuración completa del nivel
        full_config = get_level_config(level_config['level_id'])
        level_config.update(full_config)
        
        # Cargar sprites del nivel
        custom_sprites = SpriteLoader.load_level_sprites(level_config['level_id'])
    
    # Crear objetos del juego con sprites personalizados
    table = Table()
    
    # Aplicar customización de la mesa
    if level_config:
        table.table_color = level_config.get('theme', {}).get('table_color', BLACK)
        
        # Establecer sprites de porterías si están disponibles
        if 'goal_left' in custom_sprites and 'goal_right' in custom_sprites:
            table.set_goal_sprites(custom_sprites['goal_left'], custom_sprites['goal_right'])
    
    # Crear sprites con imágenes personalizadas
    puck_image = custom_sprites.get('puck', None)
    puck = Puck(custom_image=puck_image)

    human_mallet = HumanMallet(color=mallet_color)    
    
    ai_mallet_image = custom_sprites.get('mallet_ai', None)
    ai_reaction_speed = level_config.get('ai_reaction_speed', None) if level_config else None
    print(ai_reaction_speed,"esta es la reaccion")
    ai_prediction_factor = level_config.get('ai_prediction_factor', None) if level_config else None
    print(ai_prediction_factor,"esta es la prediccion")
    
    ai_mallet = AIMallet(custom_image=ai_mallet_image, 
                        reaction_speed=ai_reaction_speed,
                        prediction_factor=ai_prediction_factor)
    
    # Cargar modelo RL si se especifica
    model = None
    model_type = "original"
    
    if use_rl:
        if model_path:
            try:
                model, model_type = load_optimized_model(model_path)
                print(f"Modelo cargado: {model_path}")
            except Exception as e:
                print(f"Error cargando modelo: {e}")
                use_rl = False
        else:
            # Auto-detectar mejor modelo
            model_path, model_type = find_best_model()
            if model_path:
                try:
                    model, _ = load_optimized_model(model_path, model_type)
                    print(f"Modelo auto-detectado: {model_path}")
                except Exception as e:
                    print(f"Error cargando modelo: {e}")
                    use_rl = False
    
    # Variables del juego
    player_score = 0
    ai_score = 0
    running = True
    
    # Game over state
    game_over = False
    winner = None
    paused = False 
    waiting_for_resume_click = False  
    resume_target_pos = None 
    resume_target_radius = 20 
    
    # RL prediction management
    last_action = 4  # Default to "stay" action
    last_prediction_time = 0
    prediction_interval = 15  # ms between predictions (reducido de 20 para más responsividad)
    
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
    frame_skip = 1  # Reducido de 2 para actualizaciones más frecuentes
    
    # Physics time step control
    last_physics_update = time.time()
    fixed_physics_step = 1/120  # Ajustado para 120 FPS
    
    # For reset message
    show_reset_message = False
    reset_message_timer = 0
    
    # Button dimensions for retry
    button_width = 200
    button_height = 50
    button_x = current_width // 2 - button_width // 2
    button_y = current_height // 2 + 50
      # Loop principal del juego (adaptado del main() original)
    show_fps = False
    half_width = current_width // 2
    
    # Sprite group
    all_sprites = pygame.sprite.Group()
    all_sprites.add(human_mallet, ai_mallet, puck)
    
    while running:
        frame_start = time.time()
        
        # Get mouse position for both game and UI interaction
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        # Handle all events at once
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if not game_over: 
                        if not waiting_for_resume_click:
                            paused = not paused
                            if paused:
                                resume_target_pos = (human_mallet.position[0], human_mallet.position[1])
                elif event.key == pygame.K_f:
                    show_fps = not show_fps
                elif event.key == pygame.K_r and not game_over:
                    # Complete reset
                    puck.reset(zero_velocity=True)
                    human_mallet.position = [current_width // 4, current_height // 2]
                    human_mallet.rect.center = human_mallet.position
                    ai_mallet.position = [current_width * 3 // 4, current_height // 2]
                    ai_mallet.rect.center = ai_mallet.position
                    human_mallet.velocity = [0, 0]
                    ai_mallet.velocity = [0, 0]
                    show_reset_message = True
                    reset_message_timer = pygame.time.get_ticks()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_clicked = True

        # Only update game state if not in game over
        if not game_over and not paused and not waiting_for_resume_click:
            # Update human mallet with the mouse position
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
                            if action.ndim == 0:
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
                        if ai_mallet.position[1] > current_height * 0.75:
                            stuck_in_bottom_counter += 1
                        else:
                            stuck_in_bottom_counter = 0
                        
                        # Check if AI is stuck on the sides
                        if (ai_mallet.position[0] > current_width * 0.85 or
                            ai_mallet.position[0] < current_width * 0.55):
                            stuck_in_side_counter += 1
                        else:
                            stuck_in_side_counter = 0
                          # Calculate distances and game state
                        y_distance = abs(puck.position[1] - ai_mallet.position[1])
                        x_distance = abs(puck.position[0] - ai_mallet.position[0])
                        puck_in_ai_half = puck.position[0] > current_width // 2
                        
                        # Count recent movements
                        recent_vertical = sum(1 for a in movement_history[-10:] if a in [0, 1]) if len(movement_history) >= 10 else 0
                        recent_horizontal = sum(1 for a in movement_history[-10:] if a in [2, 3]) if len(movement_history) >= 10 else 0
                        
                        # Force movement conditions
                        force_vertical = False
                        force_horizontal = False
                        reason = ""
                        
                        # VERTICAL MOVEMENT FORCING
                        if (y_distance > force_vertical_threshold and 
                            last_vertical_move > vertical_move_cooldown and 
                            puck_in_ai_half and recent_vertical < 2):
                            force_vertical = True
                            reason = "puck_far_vertical"
                        elif stuck_in_bottom_counter > 5 and recent_vertical == 0:
                            force_vertical = True
                            reason = "stuck_in_bottom"
                        elif (len(movement_history) >= 15 and recent_vertical == 0 and 
                              puck_in_ai_half and y_distance > 40):
                            force_vertical = True
                            reason = "no_recent_vertical"
                        
                        # HORIZONTAL MOVEMENT FORCING
                        if (not force_vertical and x_distance > force_horizontal_threshold and 
                            last_horizontal_move > horizontal_move_cooldown and 
                            puck_in_ai_half and recent_horizontal < 2):
                            force_horizontal = True
                            reason = "puck_far_horizontal"
                        elif (not force_vertical and stuck_in_side_counter > 5 and recent_horizontal == 0):
                            force_horizontal = True
                            reason = "stuck_in_side"
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
                        elif force_horizontal:
                            if puck.position[0] > ai_mallet.position[0]:
                                action = 3  # Right
                            else:
                                action = 2  # Left
                            last_horizontal_move = 0
                        
                        last_action = action
                    except Exception as e:
                        print(f"Prediction error: {e}")
                    
                    last_prediction_time = current_time
                
                # Apply the action with fixed step size
                prev_position = ai_mallet.position.copy()
                move_amount = 9  # Aumentado de 7 para movimiento más rápido
                
                if last_action == 0:  # Up
                    ai_mallet.position[1] = max(ai_mallet.position[1] - move_amount, ai_mallet.radius)
                elif last_action == 1:  # Down
                    ai_mallet.position[1] = min(ai_mallet.position[1] + move_amount, current_height - ai_mallet.radius)
                elif last_action == 2:  # Left
                    ai_mallet.position[0] = max(ai_mallet.position[0] - move_amount, half_width + ai_mallet.radius)
                elif last_action == 3:  # Right
                    ai_mallet.position[0] = min(ai_mallet.position[0] + move_amount, current_width - ai_mallet.radius)
                
                # Update rect and calculate velocity
                ai_mallet.rect.center = ai_mallet.position
                ai_mallet.velocity = [
                    (ai_mallet.position[0] - prev_position[0]),
                    (ai_mallet.position[1] - prev_position[1])
                ]
            else:
                # Use simple AI behavior when RL is not active
                if not use_rl:
                    ai_mallet.update(puck.position, puck.velocity)
            
            # Fixed physics time stepping - optimizado para 120 FPS
            current_time = time.time()
            elapsed = current_time - last_physics_update
            
            # Actualizar física con timestep fijo
            accumulator = elapsed
            while accumulator >= fixed_physics_step:
                puck.update()
                accumulator -= fixed_physics_step
            
            if accumulator < elapsed:
                last_physics_update = current_time - accumulator
            
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
            
            # Check for goal collision (rebote en estructuras de porterías)
            table.check_goal_collision(puck)
            
            # Check for goals
            goal = table.is_goal(puck)
            if goal == "player":
                player_score += 1
                puck.reset("player")
                if player_score >= 7:
                    game_over = True
                    winner = "player"
                    # Stop puck and mallets when game is over
                    puck.velocity = [0, 0]
                    human_mallet.velocity = [0, 0]
                    ai_mallet.velocity = [0, 0]
            elif goal == "ai":
                ai_score += 1
                puck.reset("ai")
                if ai_score >= 7:
                    game_over = True
                    winner = "ai"
                    # Stop puck and mallets when game is over
                    puck.velocity = [0, 0]
                    human_mallet.velocity = [0, 0]
                    ai_mallet.velocity = [0, 0]
        
        # Draw everything
        # Dibujar fondo personalizado si está disponible
        has_custom_background = 'background' in custom_sprites and custom_sprites['background'] is not None
        if has_custom_background:
            screen.blit(custom_sprites['background'], (0, 0))
            # No dibujar fondo sólido en la mesa porque ya tenemos fondo personalizado
            table.draw(screen, draw_background=False, debug_mode=show_fps)
        else:
            # Si no hay fondo personalizado, dejar que la mesa dibuje su fondo
            table.draw(screen, draw_background=True, debug_mode=show_fps)
        
        # Draw glows - optimizado para mejor rendimiento
        current_fps = clock.get_fps()
        # Solo dibujar brillos si FPS es bueno o si no estamos mostrando FPS
        if not show_fps or current_fps > 50:
            # Dibujar brillos completos
            draw_glow(screen, (255, 0, 0), human_mallet.position, human_mallet.radius)
            draw_glow(screen, (0, 255, 0), ai_mallet.position, ai_mallet.radius)
            draw_glow(screen, (0, 0, 255), puck.position, puck.radius)
        
        # Draw sprites
        all_sprites.draw(screen)
        
        # Draw UI elements
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"{player_score} - {ai_score}", True, WHITE)
        screen.blit(score_text, (current_width // 2 - score_text.get_width() // 2, 20))

        if (paused or waiting_for_resume_click) and not game_over:
            # Overlay semi-transparente
            overlay = pygame.Surface((current_width, current_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (0, 0))
            if waiting_for_resume_click:
                confirm_font = pygame.font.Font(None, 36)
                confirm_text = confirm_font.render("Haz click cerca de tu mazo para reanudar", True, WHITE)
                confirm_rect = confirm_text.get_rect(center=(current_width // 2, current_height // 2 - 100))
                screen.blit(confirm_text, confirm_rect)
                
                # Dibujar área objetivo
                pygame.draw.circle(screen, (0, 255, 0, 100), resume_target_pos, resume_target_radius)
                pygame.draw.circle(screen, (0, 255, 0), resume_target_pos, resume_target_radius, 2)
                
                # Dibujar línea desde mouse hasta objetivo si está fuera
                if vector_length((mouse_pos[0]-resume_target_pos[0], mouse_pos[1]-resume_target_pos[1])) > resume_target_radius:
                    pygame.draw.line(screen, (255, 255, 255), mouse_pos, resume_target_pos, 1)
                
                # Verificar click en área
                if mouse_clicked:
                    distance = vector_length((mouse_pos[0]-resume_target_pos[0], mouse_pos[1]-resume_target_pos[1]))
                    if distance <= resume_target_radius:
                        paused = False
                        waiting_for_resume_click = False
                        resume_target_pos = None
            else:
                pause_font = pygame.font.Font(None, 72)
                pause_text = pause_font.render("PAUSA", True, WHITE)
                pause_rect = pause_text.get_rect(center=(current_width // 2, current_height // 2 - 150))
                screen.blit(pause_text, pause_rect)
                
                # Botones redondos
                button_radius = 60
                button_y = current_height // 2
                
                # Botón Reiniciar
                restart_color = (100, 100, 0) if pygame.Rect(current_width//2 - button_radius - 150, button_y - button_radius, 
                                                        button_radius*2, button_radius*2).collidepoint(mouse_pos) else (70, 70, 0)            
                restart_btn = draw_round_button(screen, restart_color, (current_width//2 - 150, button_y), button_radius, "Reiniciar", 32)

                # Botón Reanudar
                resume_color = (0, 100, 0) if pygame.Rect(current_width//2 - button_radius, button_y - button_radius, 
                                                        button_radius*2, button_radius*2).collidepoint(mouse_pos) else (0, 70, 0)
                resume_btn = draw_round_button(screen, resume_color, (current_width//2, button_y), button_radius, "Reanudar", 32)
                
                # Botón Salir
                quit_color = (100, 0, 0) if pygame.Rect(current_width//2 - button_radius + 150, button_y - button_radius, 
                                                    button_radius*2, button_radius*2).collidepoint(mouse_pos) else (70, 0, 0)
                quit_btn = draw_round_button(screen, quit_color, (current_width//2 + 150, button_y), button_radius, "Salir", 32)
                
                # Manejar clics en los botones
                if pygame.mouse.get_pressed()[0]:
                    if resume_btn.collidepoint(mouse_pos):
                            waiting_for_resume_click = True 
                    elif restart_btn.collidepoint(mouse_pos):
                        # Reset game
                        player_score = 0
                        ai_score = 0
                        game_over = False
                        winner = None
                        paused = False
                        waiting_for_resume_click = False
                        puck.reset(zero_velocity=True)
                        human_mallet.position = [current_width // 4, current_height // 2]
                        human_mallet.rect.center = human_mallet.position
                        ai_mallet.position = [current_width * 3 // 4, current_height // 2]
                        ai_mallet.rect.center = ai_mallet.position
                        human_mallet.velocity = [0, 0]
                        ai_mallet.velocity = [0, 0]

                    elif quit_btn.collidepoint(mouse_pos):
                        running = False

        # Draw game over screen if game is over
        if game_over:
            # Semi-transparent overlay
            overlay = pygame.Surface((current_width, current_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (0, 0))
            
            # Game over text
            game_over_text = "¡Has Ganado!" if winner == "player" else "¡Has Perdido!"
            game_over_surface = font.render(game_over_text, True, WHITE)
            game_over_rect = game_over_surface.get_rect(center=(current_width // 2, current_height // 2 - 100))
            screen.blit(game_over_surface, game_over_rect)
            
            # Configuración de botones
            button_radius = 60
            button_y = current_height // 2 + 50
            button_spacing = 150
            
            # Botón Reintentar (verde)
            retry_color = (0, 100, 0) if pygame.Rect(
                current_width//2 - button_spacing - button_radius, 
                button_y - button_radius, 
                button_radius*2, 
                button_radius*2
            ).collidepoint(mouse_pos) else (0, 70, 0)
            
            retry_btn = draw_round_button(
                screen, 
                retry_color, 
                (current_width//2 - button_spacing, button_y), 
                button_radius, 
                "Reintentar", 
                32
            )
            
            # Botón Salir (rojo)
            quit_color = (100, 0, 0) if pygame.Rect(
                current_width//2 + button_spacing - button_radius, 
                button_y - button_radius, 
                button_radius*2, 
                button_radius*2
            ).collidepoint(mouse_pos) else (70, 0, 0)
            
            quit_btn = draw_round_button(
                screen, 
                quit_color, 
                (current_width//2 + button_spacing, button_y), 
                button_radius, 
                "Salir", 
                32
            )
            
            # Manejar clics en los botones
            if mouse_clicked:  # Usamos la variable mouse_clicked que ya está definida
                if retry_btn.collidepoint(mouse_pos):
                    # Reset game
                    player_score = 0
                    ai_score = 0
                    game_over = False
                    winner = None
                    paused = False
                    waiting_for_resume_click = False
                    puck.reset(zero_velocity=True)
                    human_mallet.position = [current_width // 4, current_height // 2]
                    human_mallet.rect.center = human_mallet.position
                    ai_mallet.position = [current_width * 3 // 4, current_height // 2]
                    ai_mallet.rect.center = ai_mallet.position
                    human_mallet.velocity = [0, 0]
                    ai_mallet.velocity = [0, 0]
                    show_reset_message = True
                    reset_message_timer = pygame.time.get_ticks()
                
                elif quit_btn.collidepoint(mouse_pos):
                    running = False
        
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
        
        screen.blit(mode_text, (current_width // 2 - mode_text.get_width() // 2, current_height - 50))
        
        controls_text = font.render("F: FPS | ESC: Pausar", True, WHITE)
        screen.blit(controls_text, (10, current_height - 30))
        
        # Show reset message if active
        if show_reset_message:
            if pygame.time.get_ticks() - reset_message_timer < 1500:
                reset_text = font.render("¡Juego Reiniciado!", True, WHITE)
                text_rect = reset_text.get_rect(center=(current_width // 2, current_height // 2 - 50))
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
    
    # Return game result
    return {
        'victory': winner == "player" if game_over else False,
        'player_score': player_score,
        'ai_score': ai_score,
        'level_id': level_config.get('id') if level_config else None
    }