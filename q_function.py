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
    

    # def get_valid_moves(self, board, player):
    #     valid_moves = []
    #     # Simplified move generation - you'll need to implement actual checkers rules
    #     # This is just a placeholder
    #     for i in range(self.board_size):
    #         for j in range(self.board_size):
    #             if board[i][j] == player:
    #                 # Add possible moves for this piece
                    # possible_moves = self._get_piece_moves(board, i, j, player)
    #                 valid_moves.extend(possible_moves)
    #     return valid_moves
    

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
            
            return best_action or random.choice(valid_moves)
    
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
    
    
    def train(self, episodes=1000):
        for episode in range(episodes):
            board = self.initialize_board()
            done = False
            
            while not done:
                current_state = board.copy()
                valid_moves = self.env.valid_moves(player=1)
                
                if not valid_moves:
                    done = True
                    continue
                
                action = self.choose_action(current_state, valid_moves)
                previous_board = board.copy()  # Store previous board state
                new_board = self.apply_move(board, action)
                reward = self.get_reward(new_board, previous_board)  # Pass both states
                
                self.update_q_value(current_state, action, reward, new_board)
                board = new_board
                
                if self.is_game_over(board):
                    done = True


    def get_reward(self, board):
        # Example reward function
        if self.is_win(board):
            return 5.0
        elif self.is_loss(board):
            return -5.0
        elif self.is_capture_move(board):
            return 1.0
        return 0.0
    
    def is_win(self, board):
        """Check if the current player (assumed to be 1) has won"""
        # Win conditions:
        # 1. Opponent has no pieces left
        # 2. Opponent has no valid moves
        opponent_pieces = np.count_nonzero(board == 2)  # assuming 2 represents opponent
        if opponent_pieces == 0:
            return True
            
        # Check if opponent has any valid moves
        opponent_moves = self.env.valid_moves(player=2)
        return len(opponent_moves) == 0

    def is_loss(self, board):
        """Check if the current player (assumed to be 1) has lost"""
        # Loss conditions:
        # 1. Current player has no pieces left
        # 2. Current player has no valid moves
        player_pieces = np.count_nonzero(board == 1)  # assuming 1 represents current player
        if player_pieces == 0:
            return True
            
        # Check if current player has any valid moves
        player_moves = self.env.valid_moves(player=1)
        return len(player_moves) == 0

    def is_capture_move(self, board, previous_board=None):
        """
        Check if the last move was a capture move by comparing piece count
        Note: This requires keeping track of the previous board state
        """
        if previous_board is None:
            return False
            
        previous_pieces = np.count_nonzero(previous_board != 0)
        current_pieces = np.count_nonzero(board != 0)
        
        return current_pieces < previous_pieces

    def _count_pieces(self, board, player):
        """Helper method to count pieces for a specific player"""
        return np.count_nonzero(board == player)