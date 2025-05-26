# train_agent.py
import gymnasium as gym
import numpy as np
import time
import os
import pygame
import torch
from stable_baselines3 import PPO  # Switching from DQN to PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback,BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

torch.set_num_threads(6)  # Usa 4 threads CPU (ajusta según tu procesador)
torch.cuda.is_available = lambda: False

class DifficultyProgressionCallback(BaseCallback):
    """
    Callback for adjusting opponent difficulty based on agent performance.
    """
    def __init__(self, eval_env, eval_freq=50000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env  # Environment used for evaluation
        self.eval_freq = eval_freq  # How often to evaluate and adjust difficulty
        self.best_mean_reward = -float('inf')
        
    def _on_step(self):
        # Check if it's time to evaluate and adjust difficulty
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            rewards = []
            obs, _ = self.eval_env.reset()
            
            # Run a few episodes to assess performance
            for _ in range(5):  # Evaluate on 5 episodes
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = self.eval_env.step(action)
                    done = done or truncated
                    episode_reward += reward
                
                rewards.append(episode_reward)
                obs, _ = self.eval_env.reset()
            
            mean_reward = sum(rewards) / len(rewards)
            
            # Only increase difficulty if performance has improved significantly
            if mean_reward > self.best_mean_reward + 0.5:
                self.best_mean_reward = mean_reward
                
                # Access the unwrapped environment to call our method
                # This is the key fix - we need to get to the actual AirHockeyEnv instance
                unwrapped_env = self.eval_env.unwrapped
                unwrapped_env.increase_opponent_difficulty(mean_reward)
                
                if self.verbose > 0:
                    print(f"Step {self.n_calls}: Mean reward: {mean_reward:.2f}, increasing opponent difficulty")
            
            elif self.verbose > 0:
                print(f"Step {self.n_calls}: Mean reward: {mean_reward:.2f}, maintaining opponent difficulty")
                
        return True  # Continue training
    
# Import your custom environment
from ..air_hockey_env import AirHockeyEnv

def create_improved_env():
    """Create an enhanced environment with more informative observations"""
    # The base environment
    env = AirHockeyEnv()
    
    # Wrap with Monitor to track rewards and episode lengths
    env = Monitor(env)
    
    # Return wrapped environment
    return env

def train_agent(total_timesteps=500000, eval_freq=10000, model_name="air_hockey_ppo"):
    """Train a PPO agent with improved parameters for the Air Hockey environment"""
    
    # Create environment
    print("Creating training environment...")
    env = create_improved_env()
    
    # Create a separate environment for evaluation
    eval_env = create_improved_env()
    
    # # Create a directory for saving models
   
    # Crear directorios con rutas absolutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    logs_dir = os.path.join(current_dir, "logs")
    best_model_dir = os.path.join(models_dir, "best_model")
   
    
    # Asegurar que los directorios existan
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    print(f"Guardando modelos en: {models_dir}")
    print(f"Guardando logs en: {logs_dir}")
    
    # Create a checkpoint callback that saves the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=models_dir,
        name_prefix=model_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Create an evaluation callback to monitor training progress
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=None,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    # Initialize the difficulty progression callback
    difficulty_callback = DifficultyProgressionCallback(eval_env, eval_freq=25000, verbose=1)

        
    # Initialize the PPO agent with improved hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=3e-4,  # Slightly lower learning rate for more stability
        n_steps=2048,        # Collect more steps per update for better learning
        batch_size=64,
        gamma=0.99,          # Discount factor for future rewards
        gae_lambda=0.95,     # GAE parameter for advantage estimation
        clip_range=0.2,      # PPO clipping parameter
        ent_coef=0.01,       # Entropy coefficient for exploration
        vf_coef=0.5,         # Value function coefficient 
        max_grad_norm=0.5,   # Gradient clipping for stability
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])],  # Deeper network
            activation_fn=torch.nn.ReLU
        ),
        verbose=1
    )
    
    # Train the agent with callbacks
    print(f"Training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, difficulty_callback],
        progress_bar=True
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Training completed in {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds")
    
    # Save the final model
    final_model_path = f"./models/{model_name}_final"
    model.save(final_model_path)
    print(f"Final model saved as {final_model_path}")
    
    # Evaluate the trained agent
    print("Evaluating final agent...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model



def play_with_trained_model(model_name="models/air_hockey_ppo_final"):
    """Load a trained model and play in human render mode"""
    
    # Initialize pygame for rendering
    pygame.init()
    pygame.display.init()
    
    # Create the environment with human rendering
    env = AirHockeyEnv(render_mode="human", play_mode=True)  # Add a play_mode flag
    
    try:
        # Load the trained model
        from stable_baselines3 import PPO
        print(f"Loading model from {model_name}...")
        model = PPO.load(model_name)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Play a few episodes
    obs, info = env.reset()
    total_reward = 0
    clock = pygame.time.Clock()
    
    # Game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get mouse position for human player
        mouse_pos = pygame.mouse.get_pos()
        
        # Update human mallet based on mouse (left side)
        env.update_human_player_manually(mouse_pos)
        
        # Get action from model for AI mallet (right side)
        action, _states = model.predict(obs, deterministic=True)
        
        # Take step in environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Render game
        env.render()
        
        # Maintain frame rate
        clock.tick(60)
        
        # Reset if episode is done
        if done or truncated:
            print(f"Episode finished with reward {total_reward}")
            total_reward = 0
            obs, info = env.reset()
    
    # Clean up pygame
    env.close()
    
if __name__ == "__main__":
    print("Air Hockey Reinforcement Learning - Improved Training")
    print("\nOptions:")
    print("1. Train a new improved agent")
    print("2. Play with a trained agent")
    print("3. Show suggested environment improvements")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        timesteps = 500000
        try:
            timesteps_input = input(f"Number of timesteps to train (default: {timesteps}): ").strip()
            if timesteps_input:
                timesteps = int(timesteps_input)
        except ValueError:
            print(f"Invalid value, using {timesteps} timesteps")
        
        model = train_agent(total_timesteps=timesteps)
        
        play_after = input("Do you want to play with the trained model? (y/n): ").lower().startswith('y')
        if play_after:
            play_with_trained_model()
            
    elif choice == "2":
        model_path = "models/air_hockey_ppo_final"
        custom_path = input(f"Enter model path (default: {model_path}): ").strip()
        if custom_path:
            model_path = custom_path
            
        if os.path.exists(model_path + ".zip"):
            play_with_trained_model(model_path)
        else:
            print(f"No trained model found at {model_path}. Train a model first.")
            train_anyway = input("Do you want to train a new model? (y/n): ").lower().startswith('y')
            if train_anyway:
                model = train_agent()
                play_with_trained_model()
    
    elif choice == "3":
        print("\n=== Suggested Environment Improvements ===\n")
        suggested_code = modify_air_hockey_env()
        print(suggested_code)
        print("\nIncorporate these changes into your air_hockey_env.py file to improve training.")
    
    else:
        print("Invalid choice")