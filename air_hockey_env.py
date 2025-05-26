# air_hockey_env.py
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from constants import *
from sprites import Puck, HumanMallet
from table import Table

SPACE_NUMBER = 5 # 0: Up, 1: Down, 2: Left, 3: Right, 4: Stay
MAX_STEPS = 1000
MEDIUM_CAMP = WIDTH / 2
    
class AirHockeyEnv(gym.Env):
    """Custom Environment for Air Hockey game that follows gym interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None, play_mode=False):
        super(AirHockeyEnv, self).__init__()
        
        self.play_mode = play_mode
        
        self.action_space = spaces.Discrete(SPACE_NUMBER) # Define action space
        
        # Define observation space
        # [mallet_x, mallet_y, puck_x, puck_y, puck_vx, puck_vy]
        # All values are normalized between 0 and 1
       # Additional state variables for improved learning
        self.steps_since_last_hit = 0
        
        # Expand observation space to include additional features
        low = np.array([0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
            
        # Define rendering parameters
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Game objects
        self.table = Table()
        self.puck = None
        self.ai_mallet = None
        self.human_mallet = None
        self.all_sprites = None
        
        # Game state
        self.player_score = 0
        self.ai_score = 0
        self.steps = 0
        self.max_steps = MAX_STEPS
        self.opponent_skill = 0.3  # Starting skill level
        # Initialize the game
        self.reset()
    
    # Then add this method to periodically increase difficulty
    def increase_opponent_difficulty(self, current_reward):
        """Increase opponent difficulty based on AI performance"""
        # Initialize these attributes if they don't exist
        if not hasattr(self, "opponent_skill"):
            self.opponent_skill = 0.3  # Starting skill level
        if not hasattr(self, "last_average_reward"):
            self.last_average_reward = -float('inf')
            
        # Only increase difficulty if performance has improved
        if current_reward > self.last_average_reward + 0.5:
            self.opponent_skill = min(0.9, self.opponent_skill + 0.1)
            self.last_average_reward = current_reward
            print(f"Increasing opponent skill to {self.opponent_skill:.1f}")
        
        return self.opponent_skill
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game objects
        self.puck = Puck()
        self.human_mallet = HumanMallet()
        
        # Create AI mallet with custom position tracking
        self.ai_mallet_radius = 30
        self.ai_mallet_position = [WIDTH * 3 // 4, HEIGHT // 2]
        self.ai_mallet_velocity = [0, 0]
        
        # Create pygame sprite for visualization
        self.ai_mallet = pygame.sprite.Sprite()
        self.ai_mallet.image = pygame.Surface((self.ai_mallet_radius * 2, self.ai_mallet_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.ai_mallet.image, NEON_GREEN, (self.ai_mallet_radius, self.ai_mallet_radius), self.ai_mallet_radius)
        pygame.draw.circle(self.ai_mallet.image, (255, 255, 255, 150), (self.ai_mallet_radius, self.ai_mallet_radius), self.ai_mallet_radius // 2)
        self.ai_mallet.rect = self.ai_mallet.image.get_rect(center=self.ai_mallet_position)
        self.ai_mallet.mask = pygame.mask.from_surface(self.ai_mallet.image)
        
        # Group all sprites for rendering
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.human_mallet, self.ai_mallet, self.puck)
        
        # Reset game state
        self.player_score = 0
        self.ai_score = 0
        self.steps = 0
        
        # Initialize simple interactive human player behavior
        self._update_human_player()
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        self.steps += 1
        
        # Apply action to AI mallet
        prev_position = self.ai_mallet_position.copy()
        move_amount = 5
        
        # Execute action (same as before)...
        if action == 0:  # Up
            self.ai_mallet_position[1] = max(self.ai_mallet_position[1] - move_amount, self.ai_mallet_radius)
        elif action == 1:  # Down
            self.ai_mallet_position[1] = min(self.ai_mallet_position[1] + move_amount, HEIGHT - self.ai_mallet_radius)
        elif action == 2:  # Left
            self.ai_mallet_position[0] = max(self.ai_mallet_position[0] - move_amount, WIDTH // 2 + self.ai_mallet_radius)
        elif action == 3:  # Right
            self.ai_mallet_position[0] = min(self.ai_mallet_position[0] + move_amount, WIDTH - self.ai_mallet_radius)
        
        # Update AI mallet sprite position and velocity
        self.ai_mallet.rect.center = self.ai_mallet_position
        self.ai_mallet_velocity = [
            self.ai_mallet_position[0] - prev_position[0],
            self.ai_mallet_position[1] - prev_position[1]
        ]
        
        # Only update simulated human player in training mode, not play mode
        if not self.play_mode:
            self._update_human_player()
        # Action handling remains the same...
        
        # Store previous distance for reward calculation
        prev_distance = np.sqrt((self.puck.position[0] - self.ai_mallet_position[0])**2 + 
                            (self.puck.position[1] - self.ai_mallet_position[1])**2)
        
        self.puck.update() # Update puck
        
        # Check collisions
        ai_hit_puck = self._check_mallet_collision(self.ai_mallet, self.ai_mallet_position, 
                                                self.ai_mallet_radius, self.ai_mallet_velocity)
        human_hit_puck = self.puck.check_mallet_collision(self.human_mallet)
        
        # Check for goals
        goal = self.table.is_goal(self.puck)
        goal_scored = False
        
        if goal == "player":
            self.player_score += 1
            goal_scored = True
            self.puck.reset("player")
        elif goal == "ai":
            self.ai_score += 1
            goal_scored = True
            self.puck.reset("ai")
        
        # Calculate reward using improved function
        reward = self._calculate_reward(prev_distance, ai_hit_puck, goal)
        
        # Check if episode is done
        done = goal_scored or self.steps >= self.max_steps or self.player_score >= 5 or self.ai_score >= 5
        
        # Set truncated flag
        truncated = self.steps >= self.max_steps and not done
        
        # Get observation using enhanced method
        observation = self._get_observation()
        
        # Get info
        info = {
            "player_score": self.player_score,
            "ai_score": self.ai_score,
            "steps": self.steps,
            "hit_puck": ai_hit_puck
        }
        
        # Render if in human mode
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, done, truncated, info
    
    def update_human_player_manually(self, mouse_pos):
        """Allow manual control of the human player mallet during play mode"""
        if not self.play_mode:
            return
            
        # Calculate target position within bounds
        target_x = min(max(mouse_pos[0], self.human_mallet.radius), WIDTH // 2 - self.human_mallet.radius)
        target_y = min(max(mouse_pos[1], self.human_mallet.radius), HEIGHT - self.human_mallet.radius)
        
        # Update position
        self.human_mallet.position = [target_x, target_y]
        self.human_mallet.rect.center = self.human_mallet.position
        
        # Calculate velocity for physics
        self.human_mallet.velocity = [
            self.human_mallet.position[0] - self.human_mallet.prev_position[0],
            self.human_mallet.position[1] - self.human_mallet.prev_position[1]
        ]
        
        # Update previous position
        self.human_mallet.prev_position = self.human_mallet.position.copy()

    def _update_human_player(self):
        """Advanced behavior for the simulated human player during training"""
        # Define skill parameters (can be adjusted for curriculum learning)
        if not hasattr(self, "opponent_skill"):
            self.opponent_skill = 0.3  # Default starting value
        
        prediction_ability = 0.3 + (0.7 * self.opponent_skill)
        reaction_speed = 0.05 + (0.2 * self.opponent_skill)
        accuracy = 0.5 + (0.5 * self.opponent_skill)
        aggression = 0.3 + (0.6 * self.opponent_skill)
        # prediction_ability = 0.7  # How well it predicts puck trajectory (0-1)
        # reaction_speed = 0.15     # How quickly it moves to target positions (higher = faster)
        # accuracy = 0.85           # How accurately it aims at the goal (0-1)
        # aggression = 0.6          # How aggressively it pursues the puck (0-1)
        
        # Puck is on human's side - OFFENSIVE MODE
        if self.puck.position[0] < WIDTH // 2:
            # Predict where the puck will be
            predicted_y = self.puck.position[1]
            
            # If puck is moving, predict its path
            if abs(self.puck.velocity[1]) > 0.5:
                # Simple linear prediction with some error based on skill
                time_to_intercept = (self.human_mallet.position[0] - self.puck.position[0]) / max(1.0, abs(self.puck.velocity[0]))
                perfect_prediction = self.puck.position[1] + self.puck.velocity[1] * time_to_intercept
                prediction_error = (1.0 - prediction_ability) * np.random.normal(0, HEIGHT * 0.2)
                predicted_y = perfect_prediction + prediction_error
            
            # Decide whether to go for the puck or position strategically
            distance_to_puck = np.sqrt((self.puck.position[0] - self.human_mallet.position[0])**2 + 
                                (self.puck.position[1] - self.human_mallet.position[1])**2)
            
            if distance_to_puck < 150 * aggression:  # Go for the hit if close enough
                target_x = min(self.puck.position[0], WIDTH // 2 - self.human_mallet.radius)
                target_y = predicted_y
                
                # If close to puck, aim toward AI's goal with some inaccuracy
                if distance_to_puck < 50:
                    # Perfect aim would be toward middle of AI's goal
                    perfect_angle = np.arctan2(HEIGHT/2 - self.puck.position[1], 
                                            WIDTH - self.puck.position[0])
                    # Add some inaccuracy based on skill
                    angle_error = (1.0 - accuracy) * np.random.normal(0, 0.5)
                    aim_angle = perfect_angle + angle_error
                    
                    # Position mallet to hit at this angle
                    hit_offset = 30 * np.sin(aim_angle)  # Perpendicular offset for hitting
                    target_y = self.puck.position[1] + hit_offset
            else:
                # Position defensively when puck is far
                target_x = WIDTH * 0.25  # Stay back a bit
                target_y = HEIGHT / 2    # Cover the middle with slight tracking
                if self.puck.position[1] < HEIGHT * 0.3:
                    target_y = HEIGHT * 0.3
                elif self.puck.position[1] > HEIGHT * 0.7:
                    target_y = HEIGHT * 0.7
        
        # Puck is on AI's side - DEFENSIVE MODE
        else:
            # Position mallet to anticipate return
            puck_trajectory = self.puck.velocity[1] / max(0.1, abs(self.puck.velocity[0]))
            potential_y = self.puck.position[1] + puck_trajectory * (WIDTH // 2 - self.puck.position[0])
            
            # Defend based on prediction but stay within reasonable bounds
            target_x = WIDTH * 0.15  # Stay close to our goal
            target_y = np.clip(potential_y, HEIGHT * 0.2, HEIGHT * 0.8)
            
            # Add some jitter to defensive movement to be less predictable
            target_y += np.random.normal(0, HEIGHT * 0.05)
        
        # Calculate movement using current position and target
        # Limit movement to reflect human capabilities
        max_speed = 15.0  # Maximum speed the simulated human can move per frame
        
        # Calculate desired movement
        move_x = (target_x - self.human_mallet.position[0]) * reaction_speed
        move_y = (target_y - self.human_mallet.position[1]) * reaction_speed
        
        # Limit to maximum speed
        move_magnitude = np.sqrt(move_x**2 + move_y**2)
        if move_magnitude > max_speed:
            scale_factor = max_speed / move_magnitude
            move_x *= scale_factor
            move_y *= scale_factor
        
        # Update mallet position
        new_x = np.clip(self.human_mallet.position[0] + move_x, 
                    self.human_mallet.radius, 
                    WIDTH // 2 - self.human_mallet.radius)
        new_y = np.clip(self.human_mallet.position[1] + move_y, 
                    self.human_mallet.radius, 
                    HEIGHT - self.human_mallet.radius)
        
        self.human_mallet.position[0] = new_x
        self.human_mallet.position[1] = new_y
        self.human_mallet.rect.center = self.human_mallet.position
        
        # Update velocity for physics effects
        self.human_mallet.velocity = [
            move_x,
            move_y
        ]

    def _get_observation(self):
        """Return an enhanced observation with more game state information"""
        # Basic observation (normalized positions and velocities)
        basic_obs = np.array([
            self.ai_mallet_position[0] / WIDTH,
            self.ai_mallet_position[1] / HEIGHT,
            self.puck.position[0] / WIDTH,
            self.puck.position[1] / HEIGHT,
            np.clip(self.puck.velocity[0] / self.puck.max_speed, -1, 1),
            np.clip(self.puck.velocity[1] / self.puck.max_speed, -1, 1),
        ], dtype=np.float32)
        
        # Calculate additional features
        # Distance between puck and AI mallet
        puck_to_mallet_dist = np.sqrt(
            (self.puck.position[0] - self.ai_mallet_position[0])**2 + 
            (self.puck.position[1] - self.ai_mallet_position[1])**2
        ) / np.sqrt(WIDTH**2 + HEIGHT**2)  # Normalize
        
        # Distance from puck to AI goal
        puck_to_ai_goal = (WIDTH - self.puck.position[0]) / WIDTH
        
        # Distance from puck to player goal
        puck_to_player_goal = self.puck.position[0] / WIDTH
        
        # Time since last hit
        time_since_hit = min(self.steps_since_last_hit / 100.0, 1.0)  # Normalize
        
        # Is puck moving toward AI goal?
        puck_moving_to_player = 1.0 if self.puck.velocity[0] < 0 else 0.0
        
        # Enhanced observation vector
        enhanced_obs = np.append(basic_obs, [
            puck_to_mallet_dist,
            puck_to_ai_goal,
            puck_to_player_goal,
            time_since_hit,
            puck_moving_to_player,
            self.player_score / 5.0,  # Normalize by max score
            self.ai_score / 5.0,
        ])
        
        return enhanced_obs
    
    def _calculate_reward(self, prev_distance, ai_hit_puck, goal):
        """Calculate an improved reward function that encourages strategic play"""
        reward = 0.0
        
        # Current distance to puck
        current_distance = np.sqrt(
            (self.puck.position[0] - self.ai_mallet_position[0])**2 + 
            (self.puck.position[1] - self.ai_mallet_position[1])**2
        )
        
        # # Reward for being in a good defensive position when puck is in player's half
        # if self.puck.position[0] < WIDTH / 2:
        #     # Calculate ideal defensive position (adjust based on puck position)
        #     ideal_x = WIDTH * 0.75  # Defensive position
        #     ideal_y = self.puck.position[1]  # Stay aligned with puck
            
        #     # Reward for being in good defensive position
        #     defensive_distance = np.sqrt(
        #         (self.ai_mallet_position[0] - ideal_x)**2 + 
        #         (self.ai_mallet_position[1] - ideal_y)**2
        #     )
        #     defensive_reward = max(0, 0.05 * (1.0 - defensive_distance / (WIDTH/2)))
        #     reward += defensive_reward
        
        # Reward for getting closer to the puck when it's in AI's half
        if self.puck.position[0] > MEDIUM_CAMP:
            # Reward for closing distance to puck
            if current_distance < prev_distance:
                reward += 0.1 * min(1.0, (prev_distance - current_distance) * 0.1)
        
        # Reward for hitting the puck
        if ai_hit_puck:
            reward += 0.5
            
            # Calculate angle of hit relative to opponent's goal
            # Center of player's goal
            player_goal_center = [0, HEIGHT / 2]
            
            # Vector from puck to player's goal
            goal_vector = [
                player_goal_center[0] - self.puck.position[0],
                player_goal_center[1] - self.puck.position[1]
            ]
            
            # Normalize the goal vector
            goal_vector_length = np.sqrt(goal_vector[0]**2 + goal_vector[1]**2)
            if goal_vector_length > 0:
                goal_vector = [
                    goal_vector[0] / goal_vector_length,
                    goal_vector[1] / goal_vector_length
                ]
            
            # Check alignment of puck velocity with goal vector using dot product
            puck_velocity_normalized = [0, 0]
            puck_speed = np.sqrt(self.puck.velocity[0]**2 + self.puck.velocity[1]**2)
            if puck_speed > 0:
                puck_velocity_normalized = [
                    self.puck.velocity[0] / puck_speed,
                    self.puck.velocity[1] / puck_speed
                ]
            
            # Dot product gives -1 to 1 (1 means perfectly aligned)
            alignment = (goal_vector[0] * puck_velocity_normalized[0] + 
                        goal_vector[1] * puck_velocity_normalized[1])
            
            # Higher reward for better alignment with the goal
            alignment_reward = max(0, alignment)  # Only reward positive alignment
            reward += 2.0 * alignment_reward * (puck_speed / self.puck.max_speed)
            
            # Reset steps since last hit
            self.steps_since_last_hit = 0
        else:
            # Increment steps since last hit
            self.steps_since_last_hit += 1
        
        # Big rewards/penalties for goals
        if goal == "player":  # Human scored
            reward -= 5.0
        elif goal == "ai":    # AI scored
            reward += 10.0
        
        # Small penalty for excessive movement (encourages efficient motion)
        movement_penalty = 0.01 * min(1.0, np.sqrt(self.ai_mallet_velocity[0]**2 + self.ai_mallet_velocity[1]**2) / 10.0)
        reward -= movement_penalty
        
        return reward
    
    
    def _check_mallet_collision(self, mallet, mallet_position, mallet_radius, mallet_velocity):
        """Custom collision detection between puck and AI mallet"""
        # Check if centers are within sum of radii
        dx = self.puck.position[0] - mallet_position[0]
        dy = self.puck.position[1] - mallet_position[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance <= self.puck.radius + mallet_radius:
            # Normalize direction vector
            if distance > 0:
                dx /= distance
                dy /= distance
            else:
                dx, dy = 1, 0  # Default direction if centers exactly overlap
            
            # Reposition the puck outside the mallet
            self.puck.position[0] = mallet_position[0] + (mallet_radius + self.puck.radius + 1) * dx
            self.puck.position[1] = mallet_position[1] + (mallet_radius + self.puck.radius + 1) * dy
            
            # Ensure puck stays within bounds
            self.puck.position[0] = max(self.puck.radius, min(self.puck.position[0], WIDTH - self.puck.radius))
            self.puck.position[1] = max(self.puck.radius, min(self.puck.position[1], HEIGHT - self.puck.radius))
            
            # Transfer some of mallet's velocity to puck
            mallet_speed_contribution = np.sqrt(mallet_velocity[0]**2 + mallet_velocity[1]**2) * 0.5
            
            # Set puck velocity based on collision
            self.puck.velocity[0] = dx * (6 + mallet_speed_contribution)
            self.puck.velocity[1] = dy * (6 + mallet_speed_contribution)
            
            # Update puck rect
            self.puck.rect.center = self.puck.position
            return True
            
        return False
    
    def render(self):
        """Render the game state"""
        if self.render_mode is None:
            return
            
        # Initialize display if not already done
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption("Air Hockey - Gymnasium")
            else:  # rgb_array
                self.screen = pygame.Surface((WIDTH, HEIGHT))
                
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        # Handle pygame events here to keep everything in one place
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return False  # Signal to main loop to exit
        
        # Draw game elements
        self.table.draw(self.screen)
        self.all_sprites.draw(self.screen)
        
        # Draw scores
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"{self.player_score} - {self.ai_score}", True, WHITE)
        self.screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))
        
        # Update display
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        # Return RGB array if required
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
            
        return True  # Signal that everything is OK
    
    def close(self):
        """Clean up resources"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            
    