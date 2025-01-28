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
    
    def train(self, epochs, pretrained_agent=None, random_agent=None, random_agent_percentage=0.3, self_play_percentage=0.3):
        """
        Q-learning training loop with varied opponents.

        Args:
            epochs (int): Number of games to play
            pretrained_agent: Pretrained agent for training in the final phase (default: None)
            random_agent: Random agent for training in the initial phase (default: None)
            random_agent_percentage (float): Percentage of epochs to train against a random agent.
            self_play_percentage (float): Percentage of epochs to train in self-play mode.

        Returns:
            list: History of rewards per episode
        """
        rewards_history = []

        # Determine phase boundaries
        random_agent_phase = int(random_agent_percentage * epochs)
        self_play_phase = int(self_play_percentage * epochs)

        for episode in range(epochs):
            self.env.reset()
            episode_reward = 0
            done = False

            # Determine opponent based on the current phase
            if episode < self_play_percentage:
                opponent = self
            elif episode < random_agent_phase + self_play_phase and random_agent is not None:
                opponent = random_agent
            else:
                opponent = pretrained_agent

            while not done:
                current_state = np.copy(self.env.board)
                valid_moves = self.env.valid_moves(self.env.player)

                if not valid_moves:
                    break

                # Agent's turn (Player 1)
                if self.env.player == 1:
                    action = self.q_learning.choose_action(current_state, valid_moves)
                    next_state, reward, done, _ = self.env.step(action, self.env.player, self)
                    self.q_learning.update_q_value(current_state, action, reward, next_state)
                    episode_reward += reward  # Only track rewards for Player 1

                # Opponent's turn (Player -1) 
                else:
                    if opponent is not None:
                        action = (opponent.q_learning.choose_action(current_state, valid_moves) 
                                  if hasattr(opponent, "q_learning") else opponent.select_action(current_state))
                        next_state, _, done, _ = self.env.step(action, self.env.player, opponent)

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
