import numpy as np
import random
from q_function import CheckersQLearning

class LearningAgent:

    def __init__(self, step_size, epsilon, env):
        self.env = env
        # Pass the hyperparameters to CheckersQLearning
        self.q_learning = CheckersQLearning(
            learning_rate=step_size,
            epsilon=epsilon
        )


    def is_win(self, board):
        """Check if current player has won"""
        winner = self.env.game_winner(board)
        return winner == self.env.player

    def is_loss(self, board):
        """Check if current player has lost"""
        winner = self.env.game_winner(board)
        return winner == -self.env.player

    def is_capture_move(self, board):
        """Check if the last move was a capture"""
        # Since we don't have access to previous board state in this method,
        # we'll rely on the capture detection in env.step
        return False  # This will be handled by env.step's reward instead
    
    def get_reward(self, board):
        """Compute reward based on game state"""
        # Check win/loss conditions
        winner = self.env.game_winner(board)
        if winner == self.env.player:
            return 5.0
        elif winner == -self.env.player:
            return -5.0
        
        # Normal move
        return 0.0
    
    
    def learning(self, episodes=1000):
        """
        Q-learning training loop
        """
        rewards_history = []
        
        for episode in range(episodes):
            self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                current_state = self.env.board.copy()
                # Get valid moves for current player
                valid_moves = self.env.valid_moves(self.env.player)
                
                if not valid_moves:
                    break
                    
                # Select action using epsilon-greedy
                action = self.select_action(current_state)
                previous_board = current_state.copy()
                
                # Take action and observe new state and reward
                new_state, env_reward = self.env.step(action, self.env.player)
                
                # Get additional reward from our reward function
                additional_reward = self.get_reward(new_state)
                total_reward = env_reward + additional_reward
                
                # Update Q-value
                self._update_q_value(current_state, action, total_reward, new_state)
                
                episode_reward += total_reward
                
                # Check if game is over
                if self.env.game_winner(new_state) is not None:
                    done = True
                    
            rewards_history.append(episode_reward)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Average Reward: {np.mean(rewards_history[-100:]):.2f}")
                
        return rewards_history

    def _update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value for state-action pair
        """
        state_key = str(state.tolist())
        next_state_key = str(next_state.tolist())
        action_key = str(action)
        
        # Initialize Q-values if not exists
        if state_key not in self.q_learning.q_table:
            self.q_learning.q_table[state_key] = {}
        if action_key not in self.q_learning.q_table[state_key]:
            self.q_learning.q_table[state_key][action_key] = 0.0
            
        # Get maximum Q-value for next state
        next_max_q = 0
        if next_state_key in self.q_learning.q_table:
            next_max_q = max(self.q_learning.q_table[next_state_key].values(), default=0)
        
        # Q-learning update formula
        current_q = self.q_learning.q_table[state_key][action_key]
        new_q = current_q + self.q_learning.learning_rate * (
            reward + self.q_learning.discount_factor * next_max_q - current_q
        )
        self.q_learning.q_table[state_key][action_key] = new_q

    def select_action(self, state):
        """
        Choose action using epsilon-greedy strategy
        
        Args:
            state: Current board state
        """
        valid_moves = self.env.valid_moves(self.env.player)
        
        if not valid_moves:
            return None
            
        # Exploration: random move
        if random.random() < self.q_learning.epsilon:
            return random.choice(valid_moves)
            
        # Exploitation: best known move
        state_key = str(state.tolist())
        if state_key not in self.q_learning.q_table:
            self.q_learning.q_table[state_key] = {}
            
        best_value = float('-inf')
        best_action = None
        
        for move in valid_moves:
            move_key = str(move)
            if move_key not in self.q_learning.q_table[state_key]:
                self.q_learning.q_table[state_key][move_key] = 0.0
                
            if self.q_learning.q_table[state_key][move_key] > best_value:
                best_value = self.q_learning.q_table[state_key][move_key]
                best_action = move
                
        return best_action or random.choice(valid_moves)

