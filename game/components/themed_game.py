import os
import sys
import pygame
import numpy as np
import time

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from game.config.level_config import get_level_config, get_asset_path
from sprites import Puck, HumanMallet, AIMallet
from table import Table
from utils import draw_glow, vector_length, normalize_vector
from constants import WIDTH, HEIGHT, WHITE, FPS

# Import AI-related functions
from main_improved import load_optimized_model, find_best_model, create_observation_for_model

class ThemedGame:
    def __init__(self, screen, level_id, save_system=None):
        self.screen = screen
        self.level_id = level_id
        self.save_system = save_system
        self.config = get_level_config(level_id)
        
        # Load level-specific assets
        self.load_assets()
        
        # Initialize game objects with themed sprites
        self.init_game_objects()
        
        # Initialize AI components
        self.init_ai_system()
    
    def init_ai_system(self):
        """Initialize the AI system with the best available model"""
        self.use_rl = True
        self.model = None
        self.model_type = "original"
        
        try:
            print("Loading RL model...")
            model_path, model_type = find_best_model()
            
            if model_path:
                self.model, self.model_type = load_optimized_model(model_path)
                print(f"Auto-selected model: {model_path}")
                print("Model loaded successfully!")
                print(f"Model type: {self.model_type}")
                
                # Pre-warm the model
                if self.model_type == "enhanced":
                    dummy_obs = np.zeros((21,), dtype=np.float32)
                elif self.model_type == "improved":
                    dummy_obs = np.zeros((21,), dtype=np.float32)
                else:
                    dummy_obs = np.zeros((13,), dtype=np.float32)
                self.model.predict(dummy_obs, deterministic=True)
            else:
                print("No model file found")
                self.use_rl = False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.use_rl = False
            print("Falling back to simple AI")
        
        # AI control variables
        self.last_action = 4  # Default to "stay" action
        self.last_prediction_time = 0
        self.prediction_interval = 20  # ms between predictions
        
        # Behavioral correction system
        self.force_vertical_threshold = 80
        self.vertical_move_cooldown = 15
        self.last_vertical_move = 100
        self.force_horizontal_threshold = 120
        self.horizontal_move_cooldown = 15
        self.last_horizontal_move = 100
        self.movement_history = []
        self.stuck_in_bottom_counter = 0
        self.stuck_in_side_counter = 0
        
        # Frame skip for AI updates
        self.frame_count = 0
        self.frame_skip = 2
        
        # Physics time step control
        self.last_physics_update = time.time()
        self.fixed_physics_step = 1/120
    
    def load_assets(self):
        """Load all level-specific assets"""
        self.assets = {}
        theme = self.config["theme"]
        
        # Load background if exists
        bg_path = get_asset_path(self.level_id, theme["background"])
        try:
            bg = pygame.image.load(bg_path)
            self.assets["background"] = pygame.transform.scale(bg, (WIDTH, HEIGHT))
        except:
            self.assets["background"] = None
        
        # Load mallet sprite if exists
        mallet_path = get_asset_path(self.level_id, theme["mallet"])
        try:
            mallet = pygame.image.load(mallet_path)
            self.assets["mallet"] = pygame.transform.scale(mallet, (60, 60))
        except:
            self.assets["mallet"] = None
        
        # Load puck sprite if exists
        puck_path = get_asset_path(self.level_id, theme["puck"])
        try:
            puck = pygame.image.load(puck_path)
            self.assets["puck"] = pygame.transform.scale(puck, (40, 40))
        except:
            self.assets["puck"] = None
    
    def init_game_objects(self):
        """Initialize game objects with themed sprites"""
        # Create themed table
        self.table = Table()
        self.table.table_color = self.config["theme"]["table_color"]
        
        # Create themed sprites
        self.puck = Puck()
        self.human_mallet = HumanMallet()
        self.ai_mallet = AIMallet()
        
        # Apply custom sprites if available
        if self.assets["puck"]:
            self.puck.image = self.assets["puck"]
            self.puck.rect = self.puck.image.get_rect(center=self.puck.position)
        
        if self.assets["mallet"]:
            self.human_mallet.image = self.assets["mallet"]
            self.human_mallet.rect = self.human_mallet.image.get_rect(center=self.human_mallet.position)
            
            ai_mallet = self.assets["mallet"].copy()
            # Tint AI mallet to differentiate it
            ai_mallet.fill(self.config["theme"]["glow_colors"]["ai"], special_flags=pygame.BLEND_RGB_MULT)
            self.ai_mallet.image = ai_mallet
            self.ai_mallet.rect = self.ai_mallet.image.get_rect(center=self.ai_mallet.position)
    
    def update_ai(self):
        """Update AI mallet position using the RL model"""
        if self.use_rl and self.model is not None and self.frame_count == 0:
            current_time = pygame.time.get_ticks()
            
            if current_time - self.last_prediction_time > self.prediction_interval:
                observation = create_observation_for_model(
                    self.ai_mallet, self.puck, self.human_mallet, 
                    self.player_score, self.ai_score, self.model_type
                )
                
                try:
                    action, _states = self.model.predict(observation, deterministic=True)
                    if isinstance(action, np.ndarray):
                        action = int(action.item() if action.ndim == 0 else action[0])
                    else:
                        action = int(action)
                    
                    # Apply behavioral corrections
                    self.apply_behavioral_corrections(action)
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
                
                self.last_prediction_time = current_time
            
            # Apply the action
            self.apply_ai_action()
        else:
            # Use simple AI behavior when RL is not active
            if not self.use_rl:
                self.ai_mallet.update(self.puck.position)
    
    def apply_behavioral_corrections(self, action):
        """Apply behavioral corrections to the AI's actions"""
        # Implementation of behavioral correction system
        # (Copy the behavioral correction logic from main.py)
        pass
    
    def apply_ai_action(self):
        """Apply the AI's chosen action"""
        prev_position = self.ai_mallet.position.copy()
        move_amount = 7
        
        if self.last_action == 0:  # Up
            self.ai_mallet.position[1] = max(self.ai_mallet.position[1] - move_amount, self.ai_mallet.radius)
        elif self.last_action == 1:  # Down
            self.ai_mallet.position[1] = min(self.ai_mallet.position[1] + move_amount, HEIGHT - self.ai_mallet.radius)
        elif self.last_action == 2:  # Left
            self.ai_mallet.position[0] = max(self.ai_mallet.position[0] - move_amount, WIDTH//2 + self.ai_mallet.radius)
        elif self.last_action == 3:  # Right
            self.ai_mallet.position[0] = min(self.ai_mallet.position[0] + move_amount, WIDTH - self.ai_mallet.radius)
        
        # Update rect and calculate velocity
        self.ai_mallet.rect.center = self.ai_mallet.position
        self.ai_mallet.velocity = [
            (self.ai_mallet.position[0] - prev_position[0]),
            (self.ai_mallet.position[1] - prev_position[1])
        ]
    
    def draw_themed_elements(self, screen):
        """Draw themed background and table"""
        if self.assets["background"]:
            screen.blit(self.assets["background"], (0, 0))
        else:
            screen.fill(self.config["theme"]["table_color"])
        
        # Draw table elements
        self.table.draw(screen, debug_mode=self.debug_mode if hasattr(self, 'debug_mode') else False)
    
    def draw_themed_glows(self, screen):
        """Draw themed glow effects"""
        theme = self.config["theme"]["glow_colors"]
        draw_glow(screen, theme["player"], self.human_mallet.position, self.human_mallet.radius)
        draw_glow(screen, theme["ai"], self.ai_mallet.position, self.ai_mallet.radius)
        draw_glow(screen, theme["puck"], self.puck.position, self.puck.radius)
    
    def run(self):
        """Run the themed game"""
        # Create sprite group
        all_sprites = pygame.sprite.Group()
        all_sprites.add(self.human_mallet, self.ai_mallet, self.puck)
        
        # Initialize game state
        clock = pygame.time.Clock()
        running = True
        game_over = False
        winner = None
        self.player_score = 0
        self.ai_score = 0
        
        # Button dimensions for retry
        button_width = 200
        button_height = 50
        button_x = WIDTH // 2 - button_width // 2
        button_y = HEIGHT // 2 + 50
        
        while running:
            # Get mouse position for both game and UI interaction
            mouse_pos = pygame.mouse.get_pos()
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "exit"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "back_to_menu"
            
            # Update game objects if not in game over
            if not game_over:
                # Update human mallet
                self.human_mallet.update(mouse_pos)
                
                # Update AI
                self.frame_count = (self.frame_count + 1) % self.frame_skip
                self.update_ai()
                
                # Update physics
                current_time = time.time()
                elapsed = current_time - self.last_physics_update
                
                if elapsed >= self.fixed_physics_step:
                    self.puck.update()
                    self.last_physics_update = current_time
                
                # Check collisions
                if self.puck.check_mallet_collision(self.human_mallet):
                    if vector_length(self.puck.velocity) < 2:
                        direction = normalize_vector([
                            self.puck.position[0] - self.human_mallet.position[0],
                            self.puck.position[1] - self.human_mallet.position[1]
                        ])
                        self.puck.velocity[0] += direction[0] * 1
                        self.puck.velocity[1] += direction[1] * 1
                
                if self.puck.check_mallet_collision(self.ai_mallet):
                    if vector_length(self.puck.velocity) < 2:
                        direction = normalize_vector([
                            self.puck.position[0] - self.ai_mallet.position[0],
                            self.puck.position[1] - self.ai_mallet.position[1]
                        ])
                        self.puck.velocity[0] += direction[0] * 1
                        self.puck.velocity[1] += direction[1] * 1
                
                # Check for goal collision (rebote en estructuras de porterías)
                self.table.check_goal_collision(self.puck)
                
                # Check for goals
                goal = self.table.is_goal(self.puck)
                if goal == "player":
                    self.player_score += 1
                    self.puck.reset("player")
                    if self.player_score >= 7:
                        game_over = True
                        winner = "player"
                        # Stop puck and mallets
                        self.puck.velocity = [0, 0]
                        self.human_mallet.velocity = [0, 0]
                        self.ai_mallet.velocity = [0, 0]
                elif goal == "ai":
                    self.ai_score += 1
                    self.puck.reset("ai")
                    if self.ai_score >= 7:
                        game_over = True
                        winner = "ai"
                        # Stop puck and mallets
                        self.puck.velocity = [0, 0]
                        self.human_mallet.velocity = [0, 0]
                        self.ai_mallet.velocity = [0, 0]
            
            # Draw everything
            self.draw_themed_elements(self.screen)
            self.draw_themed_glows(self.screen)
            all_sprites.draw(self.screen)
            
            # Draw UI elements
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"{self.player_score} - {self.ai_score}", True, WHITE)
            self.screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))
            
            # Draw game over screen if game is over
            if game_over:
                # Semi-transparent overlay
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 128))
                self.screen.blit(overlay, (0, 0))
                
                # Game over text
                game_over_text = "¡Has Ganado!" if winner == "player" else "¡Has Perdido!"
                game_over_surface = font.render(game_over_text, True, WHITE)
                game_over_rect = game_over_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
                self.screen.blit(game_over_surface, game_over_rect)
                
                # Draw retry button
                button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
                button_color = (70, 70, 70) if button_rect.collidepoint(mouse_pos) else (50, 50, 50)
                pygame.draw.rect(self.screen, button_color, button_rect)
                pygame.draw.rect(self.screen, WHITE, button_rect, 2)
                retry_text = font.render("Reintentar", True, WHITE)
                retry_rect = retry_text.get_rect(center=(WIDTH // 2, button_y + button_height // 2))
                self.screen.blit(retry_text, retry_rect)
                
                # Check for button click
                if button_rect.collidepoint(mouse_pos) and pygame.mouse.get_pressed()[0]:
                    # Reset game
                    self.player_score = 0
                    self.ai_score = 0
                    game_over = False
                    winner = None
                    self.puck.reset(zero_velocity=True)
                    self.human_mallet.position = [WIDTH // 4, HEIGHT // 2]
                    self.human_mallet.rect.center = self.human_mallet.position
                    self.ai_mallet.position = [WIDTH * 3 // 4, HEIGHT // 2]
                    self.ai_mallet.rect.center = self.ai_mallet.position
                    self.human_mallet.velocity = [0, 0]
                    self.ai_mallet.velocity = [0, 0]
            
            # Update display
            pygame.display.flip()
            clock.tick(FPS)
        
        return "back_to_menu" 