import torch
import numpy as np
import random
from .model import CheckersDQN
from .replay_buffer import ReplayBuffer
import torch.nn.functional as F


class DQNAgent:
    def __init__(self, board_size=6, learning_rate=0.001, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, 
                 gamma=0.99, buffer_size=10000, batch_size=64):
        self.board_size = board_size
        self.n_positions = board_size * board_size
        self.n_directions = 4
        self.n_actions = self.n_positions * self.n_directions
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks
        self.policy_net = CheckersDQN(board_size).to(self.device)
        self.target_net = CheckersDQN(board_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
    def encode_state(self, board):
        """Convert board to one-hot encoded tensor"""
        # Implementation depends on your board representation
        pass
        
    def encode_action(self, move):
        """Convert move to action index"""
        from_pos, to_pos = move
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        from_idx = from_row * self.board_size + from_col
        
        # Determine direction (0: NE, 1: NW, 2: SE, 3: SW)
        if to_row < from_row:  # Moving North
            direction = 0 if to_col > from_col else 1  # NE or NW
        else:  # Moving South
            direction = 2 if to_col > from_col else 3  # SE or SW
        
        return from_idx * self.n_directions + direction
    
    def decode_action(self, action_idx):
        """Convert action index to move"""
        from_idx = action_idx // self.n_directions
        direction = action_idx % self.n_directions
        
        from_row = from_idx // self.board_size
        from_col = from_idx % self.board_size
        
        if direction == 0:  # NE
            to_row, to_col = from_row - 1, from_col + 1
        elif direction == 1:  # NW
            to_row, to_col = from_row - 1, from_col - 1
        elif direction == 2:  # SE
            to_row, to_col = from_row + 1, from_col + 1
        else:  # SW
            to_row, to_col = from_row + 1, from_col - 1
        
        return ((from_row, from_col), (to_row, to_col))
    
    def choose_action(self, state, valid_moves):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(valid_moves) if valid_moves else None
        
        state_tensor = self.encode_state(state).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        # Create mask for valid actions
        valid_action_indices = [self.encode_action(move) for move in valid_moves]
        mask = torch.zeros(self.n_actions, device=self.device)
        mask[valid_action_indices] = 1.0
        
        # Set invalid actions to very negative values
        masked_q_values = q_values.clone()
        masked_q_values[mask == 0] = float('-inf')
        
        # Choose best valid action
        action_idx = torch.argmax(masked_q_values).item()
        return self.decode_action(action_idx)
    
    def update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        self.memory.add(state, action, reward, next_state, done)
    
    def learn(self):
        """Update policy network from replay buffer"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.stack([self.encode_state(s) for s in states]).to(self.device)
        action_batch = torch.tensor([self.encode_action(a) for a in actions], 
                                   dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_state_batch = torch.stack([self.encode_state(s) for s in next_states]).to(self.device)
        done_batch = torch.tensor(dones, dtype=torch.float).to(self.device)
        
        # Compute Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_model(self, filepath):
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)
        
    def load_model(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
