import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config.params import NUM_TARGETS, NUM_OBSTACLES


class QMixer(nn.Module):
    """
    QMIX network that mixes individual agent Q-values into a global Q-value
    """
    def __init__(self, state_dim, n_agents, mixing_embed_dim=32):
        super(QMixer, self).__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim
        
        # Create first layer weights and biases
        self.w1 = nn.Linear(state_dim, mixing_embed_dim)
        self.b1 = nn.Linear(state_dim, mixing_embed_dim)
        
        # Create second layer weights and biases
        self.w2 = nn.Linear(mixing_embed_dim, 1)
        self.b2 = nn.Linear(state_dim, 1)
        
    def forward(self, agent_qs, states):
        """
        Forward pass through the mixer network
        
        Args:
            agent_qs: Individual Q-values from each agent [batch_size, n_agents]
            states: Global state inputs [batch_size, state_dim]
            
        Returns:
            Combined Q-value for the team
        """
        batch_size = agent_qs.size(0)
        
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
        
        # First layer
        w1 = torch.abs(self.w1(states))  # [batch_size, mixing_embed_dim]
        b1 = self.b1(states)             # [batch_size, mixing_embed_dim]
        
        # Reshape for hidden layer calculation
        w1 = w1.view(batch_size, self.embed_dim, 1)  # [batch_size, mixing_embed_dim, 1]
        b1 = b1.view(batch_size, 1, self.embed_dim)  # [batch_size, 1, mixing_embed_dim]
        
        # Calculate hidden layer: h = ReLU(w1 * agent_qs + b1)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # [batch_size, 1, mixing_embed_dim]
        
        # Second layer
        w2 = torch.abs(self.w2(states))  # [batch_size, 1]
        b2 = self.b2(states)             # [batch_size, 1]
        
        # Reshape for output calculation
        w2 = w2.view(batch_size, 1, 1)   # [batch_size, 1, 1]
        b2 = b2.view(batch_size, 1, 1)   # [batch_size, 1, 1]
        
        # Calculate output: q_total = w2 * h + b2
        q_total = torch.bmm(hidden, w2) + b2  # [batch_size, 1, 1]
        
        return q_total.squeeze(-1)  # [batch_size, 1]


class QNetwork(nn.Module):
    """
    Individual agent Q-networks that take local observations and output Q-values for actions
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, obs):
        """
        Forward pass through the Q-network
        
        Args:
            obs: Local observation for each agent [batch_size, obs_dim]
            
        Returns:
            Q-values for each action [batch_size, action_dim]
        """
        return self.network(obs)


class QMIX:
    """
    QMIX implementation for multi-agent reinforcement learning
    """
    def __init__(self, obs_dim, action_dim, state_dim, hidden_dim=64, lr=1e-4, gamma=0.99, device="cpu"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.device = device
        
        # Determine number of agents from the state dimension
        # Assuming state contains x,y,z coordinates for each drone
        self.n_agents = (state_dim - (NUM_TARGETS * 3) - (NUM_OBSTACLES * 3)) // 3
        
        # Create Q-networks for each agent (sharing parameters)
        self.q_network = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.target_q_network = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Create mixer networks
        self.mixer = QMixer(state_dim, self.n_agents).to(device)
        self.target_mixer = QMixer(state_dim, self.n_agents).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        # Create optimizer
        self.optimizer = optim.Adam(list(self.q_network.parameters()) + list(self.mixer.parameters()), lr=lr)
        
        # For exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def act(self, observations, epsilon=None):
        """
        Select actions for each agent based on their observations
        
        Args:
            observations: Dictionary of observations for each agent
            epsilon: Exploration parameter (if None, use self.epsilon)
            
        Returns:
            Dictionary of actions for each agent
        """
        actions = {}
        eps = epsilon if epsilon is not None else self.epsilon
        
        for agent_id, obs in observations.items():
            if np.random.random() < eps:
                # Random action
                actions[agent_id] = np.random.randint(0, self.action_dim)
            else:
                # Greedy action
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_network(obs_tensor).squeeze(0)
                actions[agent_id] = q_values.argmax().item()
                
        return actions
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_networks(self, tau=1.0):
        """
        Update target networks with polyak averaging
        
        Args:
            tau: Interpolation parameter (1.0 = hard update)
        """
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def train(self, batch):
        """
        Train the QMIX network on a batch of experiences
        
        Args:
            batch: Dictionary containing tensors for states, obs, actions, rewards, next_states, next_obs, dones
            
        Returns:
            Loss value
        """
        # For visualization purposes, we don't need to implement the full training function
        pass