import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
from config.params import *

class DroneSwarmEnv(ParallelEnv):
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.action_space = {f"drone_{i}": Discrete(6) for i in range(NUM_DRONES)}
        self.observation_space = {
            f"drone_{i}": Box(low=0, high=GRID_SIZE, shape=(9,))
            for i in range(NUM_DRONES)
        }

        self.drones = [np.random.randint(0, GRID_SIZE, 3) for _ in range(NUM_DRONES)]
        self.targets = [np.random.randint(0, GRID_SIZE, 3) for _ in range(NUM_TARGETS)]
        self.obstacles = [np.random.randint(0, GRID_SIZE, 3) for _ in range(NUM_OBSTACLES)]
        self.mapped = set()
        self.steps = 0
        # Track previous distances for shaping rewards
        self.prev_distances = self._calculate_all_distances()

    def reset(self):
        self.drones = [np.random.randint(0, GRID_SIZE, 3) for _ in range(NUM_DRONES)]
        self.targets = [np.random.randint(0, GRID_SIZE, 3) for _ in range(NUM_TARGETS)]
        self.obstacles = [np.random.randint(0, GRID_SIZE, 3) for _ in range(NUM_OBSTACLES)]
        self.mapped = set()
        self.steps = 0
        self.prev_distances = self._calculate_all_distances()
        
        # Create a proper global state and return it in info
        global_state = self._get_global_state()
        info = {
            "global_state": global_state,
            "next_global_state": global_state,  # Same at reset
            "drone_positions": self.drones,
            "obstacle_positions": self.obstacles,
            "target_positions": self.targets
        }
        
        return self._get_obs(), info  # Return the info dictionary

    def step(self, actions):
        current_global_state = self._get_global_state()
        self.steps += 1

        # Move drones based on actions
        for i, action in actions.items():
            drone = self.drones[int(i.split('_')[1])]
            if action == 0:  # Move up
                drone[1] = min(drone[1] + 1, self.grid_size - 1)
            elif action == 1:  # Move down
                drone[1] = max(drone[1] - 1, 0)
            elif action == 2:  # Move left
                drone[0] = max(drone[0] - 1, 0)
            elif action == 3:  # Move right
                drone[0] = min(drone[0] + 1, self.grid_size - 1)
            elif action == 4:  # Move forward
                drone[2] = min(drone[2] + 1, self.grid_size - 1)
            elif action == 5:  # Move backward
                drone[2] = max(drone[2] - 1, 0)

        # Handle collisions with obstacles
        for idx, drone in enumerate(self.drones):
            for obstacle in self.obstacles:
                if np.all(np.abs(drone - obstacle) < 1.5):  # More generous collision detection
                    # Apply penalty for hitting obstacle
                    rewards[f"drone_{idx}"] -= 5.0
                    # Reset position
                    self.drones[idx] = np.random.randint(0, self.grid_size, 3)

        # Calculate current distances for reward shaping
        current_distances = self._calculate_all_distances()

        # Initialize rewards
        rewards = {f"drone_{i}": 0 for i in range(len(self.drones))}
        
        # Add reward shaping based on distance changes
        for drone_idx in range(len(self.drones)):
            drone_key = f"drone_{drone_idx}"
            
            # Add small penalty for each step to encourage efficiency
            rewards[drone_key] -= 0.01
            
            # Process each target for this drone
            for target_idx in range(len(self.targets)):
                # Get previous and current distances
                prev_dist = self.prev_distances[drone_idx][target_idx]
                curr_dist = current_distances[drone_idx][target_idx]
                
                # Reward for getting closer to target
                if curr_dist < prev_dist:
                    rewards[drone_key] += 0.1 * (prev_dist - curr_dist)
                
                # Add bonus for being very close to target
                if curr_dist < 3.0:
                    rewards[drone_key] += 0.5 * (1.0 - curr_dist/3.0)
                
                # Large reward for reaching target
                if curr_dist < 1.5:
                    rewards[drone_key] += 10.0
                    print(f"Target found by drone {drone_idx}! Reward +10")

        # Update previous distances for next step
        self.prev_distances = current_distances

        # Remove targets that are reached
        targets_to_remove = []
        for target_idx, target_pos in enumerate(self.targets):
            for drone_idx, drone_pos in enumerate(self.drones):
                if np.linalg.norm(np.array(drone_pos) - np.array(target_pos)) < 1.5:
                    targets_to_remove.append(target_idx)
        
        # Remove targets from highest index to lowest to avoid index issues
        for idx in sorted(set(targets_to_remove), reverse=True):
            if idx < len(self.targets):
                self.targets.pop(idx)

        next_obs = self._get_obs()
        done = len(self.targets) == 0 or self.steps >= MAX_STEPS
        next_global_state = self._get_global_state()

        # Return comprehensive info dictionary
        return next_obs, rewards, done, {
            "global_state": current_global_state,
            "next_global_state": next_global_state,
            "drone_positions": self.drones,
            "obstacle_positions": self.obstacles,
            "target_positions": self.targets
        }

    def _calculate_all_distances(self):
        """Calculate distances from each drone to each target"""
        distances = []
        for drone_pos in self.drones:
            drone_distances = []
            for target_pos in self.targets:
                dist = np.linalg.norm(np.array(drone_pos) - np.array(target_pos))
                drone_distances.append(dist)
            # If no targets left, use high distance value
            if not drone_distances:
                drone_distances = [float('inf')]
            distances.append(drone_distances)
        return distances

    def _get_obs(self):
        obs = {}
        for i in range(NUM_DRONES):
            drone_pos = self.drones[i]
            obs[f"drone_{i}"] = np.concatenate([
                drone_pos,
                self._get_nearest_obstacle(drone_pos),
                self._get_nearest_target(drone_pos)
            ])
        return obs

    def _get_nearest_obstacle(self, drone_pos):
        min_dist = float('inf')
        closest_obstacle = np.zeros(3)
        for obstacle in self.obstacles:
            dist = np.linalg.norm(drone_pos - obstacle)
            if dist < min_dist:
                min_dist = dist
                closest_obstacle = obstacle
        return closest_obstacle

    def _get_nearest_target(self, drone_pos):
        min_dist = float('inf')
        closest_target = np.zeros(3)
        for target in self.targets:
            dist = np.linalg.norm(drone_pos - target)
            if dist < min_dist:
                min_dist = dist
                closest_target = target
        return closest_target

    def _get_global_state(self):
        """Create a flattened global state array with consistent length"""
        # Pad targets and obstacles to ensure consistent lengths
        padded_targets = self.targets + [np.zeros(3)] * (NUM_TARGETS - len(self.targets))
        padded_obstacles = self.obstacles + [np.zeros(3)] * (NUM_OBSTACLES - len(self.obstacles))
        
        # Concatenate all components
        return np.concatenate([
            np.concatenate(self.drones),
            np.concatenate(padded_targets),
            np.concatenate(padded_obstacles)
        ])