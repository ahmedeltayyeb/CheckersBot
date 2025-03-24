import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)