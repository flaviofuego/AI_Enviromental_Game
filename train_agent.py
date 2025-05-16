# train_agent.py
import gymnasium as gym
import numpy as np
import time
import os
import pygame
import torch
from stable_baselines3 import PPO  # Switching from DQN to PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import your custom environment
from air_hockey_env import AirHockeyEnv

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
    
    # Create a directory for saving models
    os.makedirs("models", exist_ok=True)
    
    # Create a checkpoint callback that saves the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix=model_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Create an evaluation callback to monitor training progress
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
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
        callback=[checkpoint_callback, eval_callback],
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
    env = AirHockeyEnv(render_mode="human")
    
    # Render once to ensure display is created
    env.render()
    
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
    done = False
    truncated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Get action from model with deterministic policy (no exploration)
        action, _states = model.predict(obs, deterministic=True)
        
        # Take step in environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Render and check if window was closed
        running = env.render()
        
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