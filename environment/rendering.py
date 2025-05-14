import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DroneSwarmVisualizer:
    def __init__(self, grid_size=10):
        """Initialize the 3D visualizer for drone swarm simulation.
        
        Args:
            grid_size (int): Size of the grid environment. Default is 10.
        """
        self.grid_size = grid_size
        
        # Initialize the 3D plot
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up the plot limits
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_zlim(0, self.grid_size)
        
        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Drone Swarm Visualization')
        
        # Store handles for plot objects
        self.drone_plots = []
        self.target_plots = []
        self.obstacle_plots = []
        
        # Show the plot
        plt.show(block=False)
    
    def update_positions(self, drones, obstacles, targets):
        """Update the positions of all objects in the visualization.
        
        Args:
            drones (list): List of drone positions as [x, y, z] arrays
            obstacles (list): List of obstacle positions as [x, y, z] arrays
            targets (list): List of target positions as [x, y, z] arrays
        """
        # Clear the plot
        self.ax.clear()
        
        # Reset the axis limits and labels
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_zlim(0, self.grid_size)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Drone Swarm Visualization')
        
        # Plot drones
        for i, drone_pos in enumerate(drones):
            self.ax.scatter(drone_pos[0], drone_pos[1], drone_pos[2], 
                           color='blue', marker='o', s=100, label=f'Drone {i}' if i == 0 else "")
        
        # Plot obstacles
        for obstacle_pos in obstacles:
            self.ax.scatter(obstacle_pos[0], obstacle_pos[1], obstacle_pos[2], 
                           color='red', marker='s', s=80, label='Obstacle' if len(self.obstacle_plots) == 0 else "")
        
        # Plot targets
        for i, target_pos in enumerate(targets):
            self.ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                           color='green', marker='^', s=100, label='Target' if i == 0 else "")
        
        # Add a legend if there are multiple objects
        if drones or obstacles or targets:
            self.ax.legend(loc='upper right')
        
        # Draw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()