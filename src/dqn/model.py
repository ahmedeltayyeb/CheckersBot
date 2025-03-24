import torch
import torch.nn as nn
import torch.nn.functional as F

class CheckersDQN(nn.Module):
    def __init__(self, board_size=6, hidden_size=128):
        super(CheckersDQN, self).__init__()
        self.board_size = board_size
        
        # Input: one-hot encoded board state
        # For 6x6 board: 36 positions × 3 possible states (empty, player1, player2)
        input_size = board_size * board_size * 3
        
        # Output: Q-value for each possible action
        # 36 positions × 4 directions
        self.n_actions = board_size * board_size * 4
        
        # Neural network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
