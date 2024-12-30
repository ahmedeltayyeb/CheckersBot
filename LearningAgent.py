import numpy as np
from q_function import CheckersQLearning

class LearningAgent:
    def __init__(self, step_size, epsilon, env):
        """
        Initialize the learning agent.
        
        Args:
            step_size (float): Learning rate for Q-learning
            epsilon (float): Exploration rate for epsilon-greedy strategy
            env: Checkers environment
        """
        self.env = env
        self.q_learning = CheckersQLearning(
            learning_rate=step_size,
            epsilon=epsilon
        )
    
    def learning(self, episodes):
        """
        Q-learning training loop.
        
        Args:
            episodes (int): Number of games to play
            
        Returns:
            list: History of rewards per episode
        """
        rewards_history = []
        
        for episode in range(episodes):
            self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                current_state = self.env.board.copy()
                valid_moves = self.env.valid_moves(self.env.player)
                
                if not valid_moves:
                    break
                
                # Select and execute action
                action = self.q_learning.choose_action(current_state, valid_moves)
                new_state, reward = self.env.step(action, self.env.player)
                
                # Update Q-values and track reward
                self.q_learning.update_q_value(current_state, action, reward, new_state)
                episode_reward += reward
                
                # Check if game is over
                if self.env.game_winner(new_state) is not None:
                    done = True
            
            rewards_history.append(episode_reward)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
        
        return rewards_history

    def select_action(self, state):
        """
        Choose action for current state.
        
        Args:
            state: Current board state
            
        Returns:
            tuple or None: Selected move coordinates or None if no valid moves
        """
        valid_moves = self.env.valid_moves(self.env.player)
        if not valid_moves:
            return None
            
        return self.q_learning.choose_action(state, valid_moves)
