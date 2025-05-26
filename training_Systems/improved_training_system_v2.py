# improved_training_system_v2.py
import gymnasium as gym
import numpy as np
import pygame
import torch
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import deque

# Import the environment and constants
from air_hockey_env import AirHockeyEnv
from constants import WIDTH, HEIGHT, MODELS, LOGS, PATH

class MovementBalanceCallback(BaseCallback):
    """Callback to monitor and encourage balanced movement during training"""
    
    def __init__(self, check_freq=25000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.action_history = deque(maxlen=10000)  # Track last 10k actions
        self.last_check = 0
        
    def _on_step(self):
        # Track the action taken
        if hasattr(self.locals, 'actions') and self.locals['actions'] is not None:
            action = self.locals['actions'][0]
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.ndim == 0 else int(action[0])
            else:
                action = int(action)
            self.action_history.append(action)
        
        # Check balance every check_freq steps
        if self.n_calls - self.last_check >= self.check_freq and len(self.action_history) > 1000:
            self.last_check = self.n_calls
            
            # Calculate action distribution
            actions = list(self.action_history)
            total = len(actions)
            
            vertical_count = sum(1 for a in actions if a in [0, 1])
            horizontal_count = sum(1 for a in actions if a in [2, 3])
            stay_count = sum(1 for a in actions if a == 4)
            
            vertical_pct = (vertical_count / total) * 100
            horizontal_pct = (horizontal_count / total) * 100
            stay_pct = (stay_count / total) * 100
            
            if self.verbose > 0:
                print(f"\nStep {self.n_calls} - Movement Analysis:")
                print(f"  Vertical (Up/Down): {vertical_pct:.1f}%")
                print(f"  Horizontal (Left/Right): {horizontal_pct:.1f}%")
                print(f"  Stay: {stay_pct:.1f}%")
                
                if vertical_pct < 15:
                    print("  ⚠️  WARNING: Low vertical movement - adjusting training")
                elif vertical_pct > 25:
                    print("  ✅ Good movement balance")
                else:
                    print("  📈 Improving movement balance")
        
        return True

class EnhancedAirHockeyEnv(AirHockeyEnv):
    """Enhanced Air Hockey environment with better movement incentives"""
    
    def __init__(self, render_mode=None, play_mode=False):
        # Initialize tracking variables before calling super().__init__
        self.movement_history = deque(maxlen=20)  # Track last 20 actions
        self.position_history = deque(maxlen=10)  # Track last 10 positions
        self.last_puck_hit_time = 0
        self.episode_step = 0
        super().__init__(render_mode, play_mode)
        
        # Override observation space to 21 dimensions for enhanced features
        import gymnasium.spaces as spaces
        low = np.array([0] * 21, dtype=np.float32)
        high = np.array([1] * 21, dtype=np.float32)
        # Set velocity bounds to -1, 1
        low[6:12] = -1.0  # Velocity components
        high[6:12] = 1.0
        # Set contextual info bounds
        low[14:17] = -1.0  # puck_in_ai_half, moving directions
        high[14:17] = 1.0
        low[18] = -1.0  # score_diff
        high[18] = 1.0
        
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
    def step(self, action):
        self.episode_step += 1
        
        # Store previous position for movement analysis
        prev_ai_position = self.ai_mallet_position.copy()
        
        # Call parent step
        observation, reward, done, truncated, info = super().step(action)
        
        # Track movement
        self.movement_history.append(action)
        self.position_history.append(self.ai_mallet_position.copy())
        
        # Calculate actual movement
        movement_distance = np.sqrt(
            (self.ai_mallet_position[0] - prev_ai_position[0])**2 + 
            (self.ai_mallet_position[1] - prev_ai_position[1])**2
        )
        
        # Enhanced reward shaping for better movement
        movement_reward = self._calculate_movement_reward(action, movement_distance)
        positioning_reward = self._calculate_positioning_reward()
        exploration_reward = self._calculate_exploration_reward()
        
        # Add rewards
        reward += movement_reward + positioning_reward + exploration_reward
        
        # Track puck hits
        if info.get('hit_puck', False):
            self.last_puck_hit_time = self.episode_step
        
        return observation, reward, done, truncated, info
    
    def _calculate_movement_reward(self, action, movement_distance):
        """Calculate rewards based on movement patterns"""
        reward = 0.0
        
        # Base movement reward - encourage any movement over staying still
        if action != 4:  # Not stay
            reward += 0.02
        
        # Vertical movement bonus - stronger incentive
        if action in [0, 1]:  # Up or Down
            reward += 0.08  # Increased from 0.05
            
            # Extra bonus if puck is far vertically
            y_distance = abs(self.puck.position[1] - self.ai_mallet_position[1])
            if y_distance > 30:
                reward += 0.05 * min(1.0, y_distance / 100.0)
        
        # Horizontal movement bonus - but less than vertical
        elif action in [2, 3]:  # Left or Right
            reward += 0.03
        
        # Penalty for excessive staying still
        if len(self.movement_history) >= 10:
            recent_actions = list(self.movement_history)[-10:]
            stay_count = sum(1 for a in recent_actions if a == 4)
            if stay_count > 7:  # More than 70% staying
                reward -= 0.05
        
        # Reward for movement diversity
        if len(self.movement_history) >= 8:
            recent_actions = list(self.movement_history)[-8:]
            unique_actions = len(set(recent_actions))
            if unique_actions >= 3:  # Using at least 3 different actions
                reward += 0.03
        
        # Penalty for repetitive horizontal-only movement
        if len(self.movement_history) >= 6:
            recent_actions = list(self.movement_history)[-6:]
            horizontal_only = all(a in [2, 3, 4] for a in recent_actions)
            if horizontal_only:
                reward -= 0.04
        
        return reward
    
    def _calculate_positioning_reward(self):
        """Reward for good positioning relative to puck"""
        reward = 0.0
        
        # Distance to puck
        distance_to_puck = np.sqrt(
            (self.puck.position[0] - self.ai_mallet_position[0])**2 + 
            (self.puck.position[1] - self.ai_mallet_position[1])**2
        )
        
        # Reward for being close to puck when it's in AI's half
        if self.puck.position[0] > WIDTH // 2:
            optimal_distance = 60  # Optimal distance to puck
            distance_reward = max(0, 0.1 * (1.0 - abs(distance_to_puck - optimal_distance) / 100.0))
            reward += distance_reward
        
        # Vertical alignment reward - encourage following puck vertically
        y_alignment = abs(self.puck.position[1] - self.ai_mallet_position[1])
        if y_alignment < 40:  # Well aligned vertically
            reward += 0.06
        elif y_alignment < 80:  # Reasonably aligned
            reward += 0.03
        
        # Defensive positioning when puck is in player's half
        if self.puck.position[0] < WIDTH // 2:
            # Reward for staying in defensive position
            ideal_x = WIDTH * 0.75
            x_distance = abs(self.ai_mallet_position[0] - ideal_x)
            if x_distance < 50:
                reward += 0.04
        
        return reward
    
    def _calculate_exploration_reward(self):
        """Reward for exploring different areas of the field"""
        reward = 0.0
        
        if len(self.position_history) >= 5:
            positions = list(self.position_history)[-5:]
            
            # Calculate position variance (exploration)
            x_positions = [pos[0] for pos in positions]
            y_positions = [pos[1] for pos in positions]
            
            x_variance = np.var(x_positions)
            y_variance = np.var(y_positions)
            
            # Reward for Y-axis exploration (vertical movement)
            if y_variance > 100:  # Good vertical exploration
                reward += 0.04
            elif y_variance > 50:  # Some vertical exploration
                reward += 0.02
            
            # Smaller reward for X-axis exploration
            if x_variance > 100:
                reward += 0.02
        
        return reward
    
    def _get_observation(self):
        """Return enhanced observation with 21 dimensions"""
        # Get basic observation from parent (13 dims)
        basic_obs = super()._get_observation()
        
        # Add enhanced features (8 more dims to make 21 total)
        # Calculate additional features
        max_velocity = 20.0
        human_vx_norm = np.clip(self.human_mallet.velocity[0] / max_velocity, -1, 1)
        human_vy_norm = np.clip(self.human_mallet.velocity[1] / max_velocity, -1, 1)
        
        # Distance from puck to human mallet
        max_distance = np.sqrt(WIDTH**2 + HEIGHT**2)
        puck_to_human_dist = np.sqrt(
            (self.puck.position[0] - self.human_mallet.position[0])**2 + 
            (self.puck.position[1] - self.human_mallet.position[1])**2
        ) / max_distance
        
        # Contextual information
        puck_in_ai_half = 1.0 if self.puck.position[0] > WIDTH // 2 else -1.0
        puck_moving_to_ai_goal = 1.0 if self.puck.velocity[0] > 0 else -1.0
        puck_moving_to_human_goal = 1.0 if self.puck.velocity[0] < 0 else -1.0
        
        # Time and state info
        time_factor = 0.5  # Placeholder
        score_diff = (self.ai_score - self.player_score) / 7.0
        
        # Enhanced observation (21 dimensions total)
        enhanced_obs = np.array([
            # Basic positions (6)
            self.ai_mallet_position[0] / WIDTH,
            self.ai_mallet_position[1] / HEIGHT,
            self.puck.position[0] / WIDTH,
            self.puck.position[1] / HEIGHT,
            self.human_mallet.position[0] / WIDTH,
            self.human_mallet.position[1] / HEIGHT,
            
            # Velocities (6)
            np.clip(self.puck.velocity[0] / 20.0, -1, 1),
            np.clip(self.puck.velocity[1] / 20.0, -1, 1),
            np.clip(self.ai_mallet_velocity[0] / 20.0, -1, 1),
            np.clip(self.ai_mallet_velocity[1] / 20.0, -1, 1),
            human_vx_norm,
            human_vy_norm,
            
            # Distances (2)
            basic_obs[6],  # puck_to_ai_dist from basic obs
            puck_to_human_dist,
            
            # Contextual info (7)
            puck_in_ai_half,
            puck_moving_to_ai_goal,
            puck_moving_to_human_goal,
            time_factor,
            score_diff,
            basic_obs[7],  # predicted_y from basic obs
            0.5  # difficulty level placeholder
        ], dtype=np.float32)
        
        return enhanced_obs
    
    def reset(self, seed=None, options=None):
        self.movement_history.clear()
        self.position_history.clear()
        self.last_puck_hit_time = 0
        self.episode_step = 0
        return super().reset(seed, options)

def create_enhanced_env():
    """Create the enhanced environment"""
    env = EnhancedAirHockeyEnv()
    env = Monitor(env)
    return env

def train_enhanced_model(timesteps=2000000, model_name="enhanced_vertical_model"):
    """Train an enhanced model with better vertical movement"""
    
    print("Creating enhanced training environment...")
    env = create_enhanced_env()
    
    # Create evaluation environment
    eval_env = create_enhanced_env()
    
    # Create directories
    models_dir = MODELS
    logs_dir = LOGS
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create callbacks
    movement_callback = MovementBalanceCallback(check_freq=25000, verbose=1)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=models_dir,
        name_prefix=model_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{models_dir}/best_{model_name}",
        log_path=logs_dir,
        eval_freq=50000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    # Enhanced PPO configuration for better exploration and learning
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=2e-4,  # Slightly lower for stability
        n_steps=2048,
        batch_size=128,  # Larger batch size
        gamma=0.995,  # Higher gamma for long-term planning
        gae_lambda=0.95,
        clip_range=0.15,  # More conservative clipping
        ent_coef=0.03,  # Higher entropy for more exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],
            activation_fn=torch.nn.ReLU,
            ortho_init=False  # Disable orthogonal initialization for better exploration
        ),
        verbose=1
    )
    
    print(f"Training enhanced model for {timesteps} timesteps...")
    print("Enhanced features:")
    print("- Strong vertical movement incentives")
    print("- Movement diversity rewards")
    print("- Position exploration bonuses")
    print("- Anti-repetition penalties")
    print("- Improved network architecture")
    
    start_time = time.time()
    
    # Train with all callbacks
    model.learn(
        total_timesteps=timesteps,
        callback=[movement_callback, checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Save the final model
    final_model_path = f"{models_dir}/{model_name}_final"
    model.save(final_model_path)
    print(f"Enhanced model saved as {final_model_path}")
    
    return model

def test_enhanced_model(model_path, num_tests=2000):
    """Test the enhanced model's movement patterns"""
    print(f"Testing enhanced model: {model_path}")
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create test environment
    env = EnhancedAirHockeyEnv()
    
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    action_names = ["Up", "Down", "Left", "Right", "Stay"]
    
    # Track movement patterns
    movement_distances = []
    vertical_movements = []
    
    obs, _ = env.reset()
    
    for i in range(num_tests):
        action, _ = model.predict(obs, deterministic=True)
        
        # Convert action to int
        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.ndim == 0 else int(action[0])
        else:
            action = int(action)
        
        action_counts[action] += 1
        
        # Track movement
        prev_pos = env.ai_mallet_position.copy()
        obs, reward, done, truncated, info = env.step(action)
        
        # Calculate movement distance
        movement_dist = np.sqrt(
            (env.ai_mallet_position[0] - prev_pos[0])**2 + 
            (env.ai_mallet_position[1] - prev_pos[1])**2
        )
        movement_distances.append(movement_dist)
        
        # Track vertical movement
        if action in [0, 1]:
            vertical_movements.append(abs(env.ai_mallet_position[1] - prev_pos[1]))
        
        if done or truncated:
            obs, _ = env.reset()
    
    # Print comprehensive results
    print(f"\n{'='*60}")
    print("ENHANCED MODEL ANALYSIS")
    print(f"{'='*60}")
    
    total = sum(action_counts.values())
    print(f"\nAction Distribution ({num_tests} steps):")
    for action_id in range(5):
        count = action_counts[action_id]
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {action_names[action_id]:>6}: {count:>4} ({pct:>5.1f}%)")
    
    vertical_actions = action_counts[0] + action_counts[1]
    horizontal_actions = action_counts[2] + action_counts[3]
    
    print(f"\nMovement Summary:")
    print(f"  Vertical: {(vertical_actions/total)*100:.1f}%")
    print(f"  Horizontal: {(horizontal_actions/total)*100:.1f}%")
    print(f"  Stay: {(action_counts[4]/total)*100:.1f}%")
    
    # Movement quality analysis
    if movement_distances:
        avg_movement = np.mean(movement_distances)
        print(f"  Average movement distance: {avg_movement:.2f} pixels")
    
    if vertical_movements:
        avg_vertical = np.mean(vertical_movements)
        print(f"  Average vertical movement: {avg_vertical:.2f} pixels")
    
    # Assessment
    print(f"\nAssessment:")
    if vertical_actions == 0:
        print("  ❌ NO VERTICAL MOVEMENT - Training failed")
    elif vertical_actions < total * 0.15:
        print("  ⚠️  Low vertical movement - Needs improvement")
    elif vertical_actions < total * 0.25:
        print("  📈 Moderate vertical movement - Good progress")
    else:
        print("  ✅ Excellent movement balance - Training successful")

def compare_models():
    """Compare different models"""
    models_to_test = [
        (f"{MODELS}/enhanced_vertical_model_final.zip", "Enhanced V2"),
        (f"{MODELS}/balanced_air_hockey_final.zip", "Balanced V1"),
        (f"{MODELS}/quick_model_final.zip", "Original Quick"),
    ]
    
    print("🔍 COMPARING MODELS")
    print("=" * 80)
    
    for model_path, model_name in models_to_test:
        if os.path.exists(model_path):
            print(f"\n📊 Testing {model_name}:")
            print("-" * 40)
            test_enhanced_model(model_path, 1000)
        else:
            print(f"\n❌ {model_name}: Model not found at {model_path}")

def main():
    print("🚀 ENHANCED VERTICAL MOVEMENT TRAINING SYSTEM V2")
    print("=" * 60)
    print("\nThis system trains models with natural vertical movement from scratch.")
    print("\nOptions:")
    print("1. Train new enhanced model (2M timesteps)")
    print("2. Train quick enhanced model (500K timesteps)")
    print("3. Test existing enhanced model")
    print("4. Compare all models")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        print("\n🎯 Training full enhanced model (2M timesteps)...")
        model = train_enhanced_model(2000000, "enhanced_vertical_model")
        
        print("\n📊 Testing the new model...")
        test_enhanced_model(f"{MODELS}/enhanced_vertical_model_final.zip")

    elif choice == "2":
        print("\n⚡ Training quick enhanced model (500K timesteps)...")
        model = train_enhanced_model(500000, "quick_enhanced_model")
        
        print("\n📊 Testing the new model...")
        test_enhanced_model(f"{MODELS}/quick_enhanced_model_final.zip")

    elif choice == "3":
        models = [
            f"{MODELS}/enhanced_vertical_model_final.zip",
            f"{MODELS}/quick_enhanced_model_final.zip",
            f"{MODELS}/balanced_air_hockey_final.zip",
        ]
        
        available = [m for m in models if os.path.exists(m)]
        
        if not available:
            print("No enhanced models found to test")
            return
        
        print("Available enhanced models:")
        for i, model in enumerate(available):
            print(f"{i+1}. {model}")
        
        try:
            idx = int(input(f"Select model (1-{len(available)}): ")) - 1
            if 0 <= idx < len(available):
                test_enhanced_model(available[idx])
        except ValueError:
            print("Invalid selection")
            
    elif choice == "4":
        compare_models()
    
    else:
        print("Invalid option")

if __name__ == "__main__":
    main() 