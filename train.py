import numpy as np
import torch
from environment.drone_env import DroneSwarmEnv
from environment.rendering import DroneSwarmVisualizer
from marl.qmix import QMIX
from marl.replay_buffer import ReplayBuffer
from config.params import *
from utils.metrics import MetricTracker

def train(num_episodes=1000):
    env = DroneSwarmEnv()
    obs_dim = env.observation_space["drone_0"].shape[0]
    act_dim = env.action_space["drone_0"].n
    
    # Calculate state_dim from environment
    state_dim = (NUM_DRONES * 3) + (NUM_TARGETS * 3) + (NUM_OBSTACLES * 3)
    
    # Initialize QMIX agent
    qmix = QMIX(obs_dim, act_dim, state_dim)
    buffer = ReplayBuffer(BUFFER_SIZE)
    metrics = MetricTracker()
    epsilon = EPSILON_START
    episode = 0
    
    try:
        while episode < num_episodes:
            obs, info = env.reset()  # Get initial observation and info
            # Verify info contains global state
            if "global_state" not in info:
                print("Warning: global_state not in info from reset()")
                # Handle missing state
                info["global_state"] = np.zeros(state_dim)
                info["next_global_state"] = np.zeros(state_dim)
            
            current_info = info
            episode_rewards = {f"drone_{i}": 0 for i in range(NUM_DRONES)}
            done = False
            step_count = 0
            
            while not done:
                actions = qmix.act(obs, epsilon)
                next_obs, rewards, done, next_info = env.step(actions)
                
                # Verify next_info contains necessary information
                if "global_state" not in next_info:
                    print("Warning: global_state not in info from step()")
                    # Handle missing state
                    next_info["global_state"] = np.zeros(state_dim)
                if "next_global_state" not in next_info:
                    print("Warning: next_global_state not in info from step()")
                    next_info["next_global_state"] = np.zeros(state_dim)
                
                # Store transition with properly formatted info
                transition = {
                    "obs": {k: v.astype(np.float32) for k, v in obs.items()},
                    "actions": {k: np.int32(v) for k, v in actions.items()},
                    "rewards": {k: float(v) for k, v in rewards.items()},
                    "next_obs": {k: v.astype(np.float32) for k, v in next_obs.items()},
                    "done": np.float32(done),
                    "info": current_info,
                    "next_info": next_info
                }
                
                buffer.add(transition)
                obs = next_obs
                current_info = next_info  # Update for next step
                
                for k in rewards:
                    episode_rewards[k] += rewards[k]
                
                # Update network if buffer is large enough
                if len(buffer) > BATCH_SIZE:
                    batch = buffer.sample(BATCH_SIZE)
                    try:
                        processed_batch = {
                            'obs': batch['obs'],
                            'actions': batch['actions'],
                            'rewards': batch['rewards'],
                            'next_obs': batch['next_obs'],
                            'done': np.array(batch['done'], dtype=np.float32),
                            'info': batch['info'],
                            'next_info': batch['next_info']
                        }
                        loss = qmix.update(processed_batch)
                    except Exception as e:
                        print(f"Batch processing error: {str(e)}")
                        continue
                
                step_count += 1
                if step_count >= MAX_STEPS:
                    break
            
            # Update epsilon and metrics
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            metrics.update(episode_rewards)
            
            if episode % TARGET_UPDATE == 0:
                qmix.update_target()
            
            if episode % 10 == 0:
                print(f"Episode {episode} | Avg Reward: {metrics.avg_reward()} | Epsilon: {epsilon:.2f}")
            
            episode += 1
            
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {str(e)}")
    
    # Save model
    try:
        torch.save(qmix.q_network.state_dict(), "models/qmix_final.pt")
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

if __name__ == "__main__":
    train()