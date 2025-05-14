import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append({
            'obs': transition['obs'],
            'actions': transition['actions'],
            'rewards': transition['rewards'],
            'next_obs': transition['next_obs'],
            'done': transition['done'],
            'info': transition['info'],
            'next_info': transition['next_info']  
        })

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return {
            'obs': [t['obs'] for t in batch],
            'actions': [t['actions'] for t in batch],
            'rewards': [t['rewards'] for t in batch],
            'next_obs': [t['next_obs'] for t in batch],
            'done': np.array([t['done'] for t in batch], dtype=np.float32),
            'info': [t['info'] for t in batch],
            'next_info': [t['next_info'] for t in batch]
        }

    def __len__(self):
        return len(self.buffer)
