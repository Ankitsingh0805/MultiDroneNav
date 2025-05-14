import time
import numpy as np
from environment.drone_env import DroneSwarmEnv
from environment.rendering import DroneSwarmVisualizer
from marl.qmix import QMIX
import torch
from config.params import *

def visualize():
    # Initialize environment
    env = DroneSwarmEnv()
    
    # Initialize visualizer with grid size
    vis = DroneSwarmVisualizer(grid_size=GRID_SIZE)
    
    # Calculate correct dimensions
    obs_dim = env.observation_space["drone_0"].shape[0]
    act_dim = env.action_space["drone_0"].n
    state_dim = (NUM_DRONES * 3) + (NUM_TARGETS * 3) + (NUM_OBSTACLES * 3)
    
    # Initialize QMIX with correct parameters
    qmix = QMIX(obs_dim, act_dim, state_dim)
    
    try:
        # Load trained model
        qmix.q_network.load_state_dict(torch.load("models/qmix_final.pt"))
        print("Loaded trained model successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing with untrained model")
    
    # Reset environment and get initial observation
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    # Initial visualization
    vis.update_positions(env.drones, env.obstacles, env.targets)
    
    print("Starting visualization. Press Ctrl+C to exit.")
    try:
        while not done and step_count < MAX_STEPS:
            # Get action using the trained policy
            actions = qmix.act(obs, epsilon=0.01)
            
            # Take step in environment
            next_obs, rewards, done, info = env.step(actions)
            
            # Calculate total reward
            episode_reward = sum(rewards.values())
            total_reward += episode_reward
            
            # Update visualization with current positions
            vis.update_positions(env.drones, env.obstacles, env.targets)
            
            # Print step information
            print(f"Step {step_count} | Reward: {episode_reward:.2f} | Total: {total_reward:.2f}")
            
            # Update for next iteration
            obs = next_obs
            step_count += 1
            
            # Add a small delay to make visualization visible
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Visualization stopped by user")
    
    print(f"Visualization completed. Total steps: {step_count}, Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    visualize()