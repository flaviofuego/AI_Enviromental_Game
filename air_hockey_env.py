# air_hockey_env.py
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from constants import *
from sprites import Puck, HumanMallet
from table import Table

class AirHockeyEnv(gym.Env):
    """Custom Environment for Air Hockey game that follows gym interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None):
        super(AirHockeyEnv, self).__init__()
        
        # Define action space
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Stay
        self.action_space = spaces.Discrete(5)
        
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
        self.max_steps = 1000
        
        # Initialize the game
        self.reset()
        
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
        
        # Action handling remains the same...
        
        # Update AI mallet sprite position and velocity
        self.ai_mallet.rect.center = self.ai_mallet_position
        self.ai_mallet_velocity = [
            self.ai_mallet_position[0] - prev_position[0],
            self.ai_mallet_position[1] - prev_position[1]
        ]
        
        # Update human player (simple AI for training)
        self._update_human_player()
        
        # Store previous distance for reward calculation
        prev_distance = np.sqrt((self.puck.position[0] - self.ai_mallet_position[0])**2 + 
                            (self.puck.position[1] - self.ai_mallet_position[1])**2)
        
        # Update puck
        self.puck.update()
        
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
    
    
    def _update_human_player(self):
        """Simple automated behavior for the human player during training"""
        if self.puck.position[0] < WIDTH // 2:
            target_y = self.puck.position[1] + np.random.randint(-30, 30)
            self.human_mallet.position[1] += (target_y - self.human_mallet.position[1]) * 0.1
            self.human_mallet.rect.center = self.human_mallet.position
    
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
        
        # Reward for being in a good defensive position when puck is in player's half
        if self.puck.position[0] < WIDTH / 2:
            # Calculate ideal defensive position (adjust based on puck position)
            ideal_x = WIDTH * 0.75  # Defensive position
            ideal_y = self.puck.position[1]  # Stay aligned with puck
            
            # Reward for being in good defensive position
            defensive_distance = np.sqrt(
                (self.ai_mallet_position[0] - ideal_x)**2 + 
                (self.ai_mallet_position[1] - ideal_y)**2
            )
            defensive_reward = max(0, 0.05 * (1.0 - defensive_distance / (WIDTH/2)))
            reward += defensive_reward
        
        # Reward for getting closer to the puck when it's in AI's half
        elif self.puck.position[0] > WIDTH / 2:
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
            
    