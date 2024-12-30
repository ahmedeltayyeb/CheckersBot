import numpy as np
import random

class CheckersQLearning:
    def __init__(self, learning_rate, epsilon, discount_factor=0.9):
        self.board_size = 6
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        
    def get_state_key(self, board):
        # Convert board state to a string for dictionary key
        return str(board.tolist())
    

    def choose_action(self, state, valid_moves):
        if random.random() < self.epsilon:
            # Exploration: choose random action
            return random.choice(valid_moves) if valid_moves else None
        else:
            # Exploitation: choose best known action
            state_key = self.get_state_key(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            # Find action with maximum Q-value
            best_value = float('-inf')
            best_action = None
            
            for move in valid_moves:
                move_key = str(move)
                if move_key not in self.q_table[state_key]:
                    self.q_table[state_key][move_key] = 0.0
                
                if self.q_table[state_key][move_key] > best_value:
                    best_value = self.q_table[state_key][move_key]
                    best_action = move
            if best_action:
                return best_action
            else:
                return random.choice(valid_moves)
    

    def update_q_value(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        action_key = str(action)
        
        # Initialize Q-values if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
            
        # Get maximum Q-value for next state
        next_max_q = 0
        if next_state_key in self.q_table:
            next_max_q = max(self.q_table[next_state_key].values(), default=0)
        
        # Q-learning update formula
        current_q = self.q_table[state_key][action_key]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[state_key][action_key] = new_q
    
    
    def train(self, env, episodes=1000):
        rewards_per_episode = []
        
        for episode in range(episodes):
            # Use env to initialize and manage game state
            state = env.reset()  # Should return initial board state
            done = False
            episode_reward = 0
            
            # Decay epsilon over time
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            while not done:
                current_state = state.copy()
                valid_moves = env.valid_moves(env.player)
                
                if not valid_moves:
                    done = True
                    continue
                
                # Choose action using our Q-learning logic
                action = self.choose_action(current_state, valid_moves)
                
                # Let environment handle move execution and reward calculation
                next_state, reward, done, _ = env.step(action)
                
                # Update Q-values
                self.update_q_value(current_state, action, reward, next_state)
                state = next_state
                episode_reward += reward
            
            rewards_per_episode.append(episode_reward)
            
            # Log progress
            if (episode + 1) % 100 == 0:
                avg_reward = sum(rewards_per_episode[-100:]) / 100
                print(f"Episode {episode + 1}/{episodes}, "
                    f"Average Reward: {avg_reward:.2f}, "
                    f"Epsilon: {self.epsilon:.3f}")
        
        return rewards_per_episode

