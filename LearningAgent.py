import numpy as np
from q_function import CheckersQLearning
from config import epsilon_decay

class LearningAgent:
    def __init__(self, learning_rate, epsilon, discount_factor, env):
        """
        Initialize the learning agent.
        
        Args:
            learning_rate (float): Learning rate for Q-learning
            epsilon (float): Exploration rate for epsilon-greedy strategy
            env: Checkers environment
        """
        self.env = env
        self.q_learning = CheckersQLearning(
            learning_rate=learning_rate,
            epsilon=epsilon,
            discount_factor=discount_factor
        )
    
    def train(self, episodes):
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
                current_state = np.copy(self.env.board)
                valid_moves = self.env.valid_moves(self.env.player)

                if not valid_moves:
                    break

                # Select and execute action
                action = self.q_learning.choose_action(current_state, valid_moves)
                next_state, reward, done, _ = self.env.step(action, self.env.player, self)

                # Update Q-values and track reward
                self.q_learning.update_q_value(current_state, action, reward, next_state)
                episode_reward += reward

            rewards_history.append(episode_reward)

            # Epsilon decay
            self.q_learning.epsilon = max(0.01, self.q_learning.epsilon * epsilon_decay)

            # Log progress
            if episode % 100 == 0 and episode != 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.q_learning.epsilon:.3f}")

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

    def choose_capture(self, captures, board):
        """
        Delegate capture choice to Q-learning agent
        """
        return self.q_learning.choose_capture(captures, board)
