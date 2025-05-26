# fix_vertical_movement.py
import gymnasium as gym
import numpy as np
import pygame
import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Import the environment
from air_hockey_env import AirHockeyEnv

class BalancedMovementCallback(BaseCallback):
    """Callback to monitor and encourage balanced movement during training"""
    
    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.last_check = 0
        
    def _on_step(self):
        # Track the action taken
        if hasattr(self.locals, 'actions') and self.locals['actions'] is not None:
            action = self.locals['actions'][0]
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.ndim == 0 else int(action[0])
            else:
                action = int(action)
            self.action_counts[action] += 1
        
        # Check balance every check_freq steps
        if self.n_calls - self.last_check >= self.check_freq:
            self.last_check = self.n_calls
            total_actions = sum(self.action_counts.values())
            
            if total_actions > 0:
                vertical_actions = self.action_counts[0] + self.action_counts[1]  # Up + Down
                horizontal_actions = self.action_counts[2] + self.action_counts[3]  # Left + Right
                
                vertical_pct = (vertical_actions / total_actions) * 100
                horizontal_pct = (horizontal_actions / total_actions) * 100
                
                if self.verbose > 0:
                    print(f"\nStep {self.n_calls} - Action Balance:")
                    print(f"  Vertical (Up/Down): {vertical_pct:.1f}%")
                    print(f"  Horizontal (Left/Right): {horizontal_pct:.1f}%")
                    print(f"  Stay: {(self.action_counts[4]/total_actions)*100:.1f}%")
                    
                    if vertical_pct < 10:
                        print("  ⚠️  WARNING: Very low vertical movement!")
                    elif vertical_pct > 40:
                        print("  ✅ Good vertical movement balance")
        
        return True

class BalancedAirHockeyEnv(AirHockeyEnv):
    """Modified Air Hockey environment that encourages balanced movement"""
    
    def __init__(self, render_mode=None, play_mode=False):
        super().__init__(render_mode, play_mode)
        self.last_ai_position = None
        self.movement_history = []
        self.vertical_movement_bonus = 0.0
        
    def step(self, action):
        # Store previous position
        if self.last_ai_position is None:
            self.last_ai_position = self.ai_mallet_position.copy()
        
        # Call parent step
        observation, reward, done, truncated, info = super().step(action)
        
        # Calculate movement direction
        movement = [
            self.ai_mallet_position[0] - self.last_ai_position[0],
            self.ai_mallet_position[1] - self.last_ai_position[1]
        ]
        
        # Track movement history (last 10 moves)
        self.movement_history.append(action)
        if len(self.movement_history) > 10:
            self.movement_history.pop(0)
        
        # Bonus for using vertical movement
        if action in [0, 1]:  # Up or Down
            reward += 0.05  # Small bonus for vertical movement
            
        # Penalty for only using horizontal movement
        if len(self.movement_history) >= 5:
            recent_actions = self.movement_history[-5:]
            horizontal_only = all(a in [2, 3, 4] for a in recent_actions)
            if horizontal_only and action in [2, 3]:
                reward -= 0.02  # Small penalty for horizontal-only movement
        
        # Bonus for balanced movement patterns
        if len(self.movement_history) >= 10:
            vertical_count = sum(1 for a in self.movement_history if a in [0, 1])
            if vertical_count >= 2:  # At least 2 vertical moves in last 10
                reward += 0.03
        
        # Enhanced reward for strategic vertical positioning
        puck_y_diff = abs(self.puck.position[1] - self.ai_mallet_position[1])
        if puck_y_diff < 50 and action in [0, 1]:  # Vertical move that aligns with puck
            reward += 0.1
        
        # Update last position
        self.last_ai_position = self.ai_mallet_position.copy()
        
        return observation, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        self.last_ai_position = None
        self.movement_history = []
        self.vertical_movement_bonus = 0.0
        return super().reset(seed, options)

def train_balanced_model(timesteps=1000000, model_name="balanced_air_hockey"):
    """Train a model with balanced movement encouragement"""
    
    print("Creating balanced training environment...")
    env = BalancedAirHockeyEnv()
    env = Monitor(env)
    
    # Create directories
    models_dir = "improved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create callback to monitor movement balance
    balance_callback = BalancedMovementCallback(check_freq=25000, verbose=1)
    
    # Initialize PPO with modified hyperparameters to encourage exploration
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Increased entropy for more exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],  # Deeper network
            activation_fn=torch.nn.ReLU
        ),
        verbose=1
    )
    
    print(f"Training balanced model for {timesteps} timesteps...")
    
    # Train with balance monitoring
    model.learn(
        total_timesteps=timesteps,
        callback=balance_callback,
        progress_bar=True
    )
    
    # Save the model
    model_path = f"{models_dir}/{model_name}_final"
    model.save(model_path)
    print(f"Balanced model saved as {model_path}")
    
    return model

def test_model_balance(model_path, num_tests=1000):
    """Test a model's action balance"""
    print(f"Testing model balance: {model_path}")
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create test environment
    env = BalancedAirHockeyEnv()
    
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    action_names = ["Up", "Down", "Left", "Right", "Stay"]
    
    obs, _ = env.reset()
    
    for i in range(num_tests):
        action, _ = model.predict(obs, deterministic=True)
        
        # Convert action to int
        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.ndim == 0 else int(action[0])
        else:
            action = int(action)
        
        action_counts[action] += 1
        
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            obs, _ = env.reset()
    
    # Print results
    print(f"\nAction Distribution ({num_tests} steps):")
    total = sum(action_counts.values())
    for action_id in range(5):
        count = action_counts[action_id]
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {action_names[action_id]:>6}: {count:>4} ({pct:>5.1f}%)")
    
    vertical_actions = action_counts[0] + action_counts[1]
    horizontal_actions = action_counts[2] + action_counts[3]
    
    print(f"\nMovement Analysis:")
    print(f"  Vertical: {(vertical_actions/total)*100:.1f}%")
    print(f"  Horizontal: {(horizontal_actions/total)*100:.1f}%")
    print(f"  Stay: {(action_counts[4]/total)*100:.1f}%")
    
    if vertical_actions == 0:
        print("  ❌ NO VERTICAL MOVEMENT")
    elif vertical_actions < horizontal_actions * 0.2:
        print("  ⚠️  Low vertical movement")
    else:
        print("  ✅ Balanced movement")

def main():
    print("🔧 VERTICAL MOVEMENT FIX")
    print("=" * 50)
    print("\nThis script addresses the issue where models only move horizontally.")
    print("\nOptions:")
    print("1. Train a new balanced model")
    print("2. Test existing model balance")
    print("3. Compare models")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        timesteps = input("Training timesteps (default: 1000000): ").strip()
        timesteps = int(timesteps) if timesteps else 1000000
        
        print(f"\nTraining balanced model for {timesteps} timesteps...")
        print("This model will be trained with:")
        print("- Bonuses for vertical movement")
        print("- Penalties for horizontal-only patterns")
        print("- Enhanced exploration (higher entropy)")
        print("- Movement balance monitoring")
        
        model = train_balanced_model(timesteps, "balanced_air_hockey")
        
        print("\nTesting the new model...")
        test_model_balance(f"improved_models/balanced_air_hockey_final.zip")
        
    elif choice == "2":
        models = [
            "improved_models/quick_model_final.zip",
            "improved_models/improved_air_hockey_final.zip",
            "models/air_hockey_ppo_final.zip"
        ]
        
        available = [m for m in models if os.path.exists(m)]
        
        if not available:
            print("No models found to test")
            return
        
        print("Available models:")
        for i, model in enumerate(available):
            print(f"{i+1}. {model}")
        
        try:
            idx = int(input(f"Select model (1-{len(available)}): ")) - 1
            if 0 <= idx < len(available):
                test_model_balance(available[idx])
        except ValueError:
            print("Invalid selection")
            
    elif choice == "3":
        print("\nComparing all available models...")
        models = [
            "improved_models/quick_model_final.zip",
            "improved_models/improved_air_hockey_final.zip", 
            "models/air_hockey_ppo_final.zip"
        ]
        
        for model_path in models:
            if os.path.exists(model_path):
                print(f"\n{'='*60}")
                test_model_balance(model_path, 500)
    
    else:
        print("Invalid option")

if __name__ == "__main__":
    main() 