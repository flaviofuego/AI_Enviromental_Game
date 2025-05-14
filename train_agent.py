# train_agent.py
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import os
import pygame  # Add this import line

from air_hockey_env import AirHockeyEnv
def train_agent(total_timesteps=50000, eval_episodes=10, model_name="air_hockey_dqn"):
    """Train a DQN agent for the Air Hockey environment"""
    
    # Create the environment
    env = AirHockeyEnv()
    
    # Initialize the agent
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        verbose=1
    )
    
    # Train the agent
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save the trained model
    model.save(model_name)
    print(f"Model saved as {model_name}")
    
    # Evaluate the agent
    print("Evaluating agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model


def play_with_trained_model(model_name="air_hockey_dqn"):
    """Load a trained model and play in human render mode"""
    
    # Initialize pygame first
    pygame.init()
    pygame.display.init()
    
    # Create the environment with human rendering
    env = AirHockeyEnv(render_mode="human")
    
    # Render once to ensure display is created
    env.render()
    
    # Load the trained model
    model = DQN.load(model_name)
    
    # Play a few episodes
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Get action from model
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
           
           
            
if __name__ == "__main__":
    print("Air Hockey Reinforcement Learning with Gymnasium and Stable-Baselines3")
    print("\nOptions:")
    print("1. Train a new agent")
    print("2. Play with a trained agent")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == "1":
        timesteps = 50000
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
        model_path = "air_hockey_dqn.zip"  # Stable-baselines3 adds .zip extension
        
        if os.path.exists(model_path):
            play_with_trained_model()
        else:
            print(f"No trained model found at {model_path}. Train a model first.")
            train_anyway = input("Do you want to train a new model? (y/n): ").lower().startswith('y')
            if train_anyway:
                model = train_agent()
                play_with_trained_model()
    else:
        print("Invalid choice")