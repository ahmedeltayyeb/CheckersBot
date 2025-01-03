import numpy as np

class checkers_env:
    def __init__(self, board=None, player=None):
        self.board = self.initialize_board() if board is None else board
        self.player = player
        # Accumulates the reward of the current game
        self.count = 0


    def initialize_board(self):
        board = np.array([[1, 0, 1, 0, 1, 0],
                      [0, 1, 0, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [-1, 0, -1, 0, -1, 0],
                      [0, -1, 0, -1, 0, -1]])
        return board


    def reset(self):
        self.board = self.initialize_board()
        self.player = 1
        self.count = 0


    def possible_pieces(self, player):
        """
        Find all pieces that belong to the current player
        
        Args:
            player: 1 for player 1, -1 for player 2
            
        Returns:
            list of tuples: Coordinates (x,y) of all pieces belonging to the player
        """
        pieces = []
        for i in range(6):  # 6x6 board
            for j in range(6):
                if self.board[i][j] == player or self.board[i][j] == player * 2:
                    pieces.append((i, j))
        return pieces


    def valid_moves(self, player):
        '''
        Normal pieces can only move forward, king pieces can move both ways:
        - Player 1 (P/K): regular pieces move downward, kings move both ways
        - Player -1 (E/Q): regular pieces move upward, kings move both ways
        '''
        def is_valid_position(x, y):
            return 0 <= x < 6 and 0 <= y < 6 and self.board[x][y] == 0

        def is_king(x, y):
            return abs(self.board[x][y]) == 2

        actions = []
        starters = self.possible_pieces(player)
        
        # Define forward and backward directions based on player
        if player == 1:
            forward_directions = [(1, -1), (1, 1)]     # Down-left, Down-right
            backward_directions = [(-1, -1), (-1, 1)]  # Up-left, Up-right
        else:  # player == -1
            forward_directions = [(-1, -1), (-1, 1)]   # Up-left, Up-right
            backward_directions = [(1, -1), (1, 1)]    # Down-left, Down-right

        for x, y in starters:
            # Regular pieces can only move forward, kings can move both ways
            directions = forward_directions
            if is_king(x, y):
                directions = forward_directions + backward_directions

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 6 and 0 <= ny < 6:  # Only check bounds here
                    if self.board[nx][ny] == 0:
                        actions.append([x, y, nx, ny])  # Regular move
                    elif self.board[nx][ny] == -player or self.board[nx][ny] == -player * 2:
                        jx, jy = x + 2*dx, y + 2*dy
                        if is_valid_position(jx, jy):   # Check both bounds and emptiness for capture landing
                            actions.append([x, y, jx, jy])  # Capture move
        return actions


    def capture_piece(self, action):
        '''
        Assign 0 to the positions of captured pieces and return True if a capture is made.
        '''
        start_row, start_col, end_row, end_col = map(int, action)
        if abs(end_row - start_row) == 2:  # Jump move
            captured_row, captured_col = (start_row + end_row) // 2, (start_col + end_col) // 2
            if self.board[captured_row, captured_col] == -self.player or self.board[captured_row, captured_col] == -self.player * 2:
                self.board[captured_row, captured_col] = 0
                return True
        return False

    
    def game_winner(self, board):
        # Check if player -1 has no pieces
        if np.sum(board < 0) == 0:
            return 1
        # Check if player 1 has no pieces
        if np.sum(board > 0) == 0:
            return -1
        
        # If neither can move, it's a draw or a win by piece count
        if len(self.valid_moves(1)) == 0 and len(self.valid_moves(-1)) == 0:
            if np.sum(board > 0) > np.sum(board < 0):
                return 1
            elif np.sum(board > 0) < np.sum(board < 0):
                return -1
            else:
                return 0  # Draw
        
        return None  # Game ongoing


    def step(self, action, player, agent):
        '''
        The transition of board and incurred reward after player performs an action.
        '''
        reward = 0
        row1, col1, row2, col2 = map(int, action)

        if action in self.valid_moves(player):
            # Store the current piece (including its king status if it is one)
            current_piece = self.board[row1, col1]
            self.board[row1, col1] = 0

            # If it's already a king, keep it as a king
            # if abs(current_piece) == 2:
            #     self.board[row2, col2] = current_piece
            # If it's a regular piece reaching the opposite end, promote to king
            if (player == 1 and row2 == 5) or (player == -1 and row2 == 0):
                self.board[row2, col2] = player * 2  # Convert to king (2 or -2)
            else:
                self.board[row2, col2] = current_piece

            if self.capture_piece(action):
                reward += 3
                # Handle chain captures
                row2, col2 = action[2], action[3]
                current_piece = self.board[row2, col2]
                
                while True:
                    captures = [
                        move for move in self.valid_moves(player)
                        if move[0] == row2 and move[1] == col2 and abs(move[2] - row2) == 2
                    ]
                    
                    if not captures:
                        break
                        
                    # Execute next capture

                    next_capture = agent.choose_capture(captures, self.board)
                    self.board[row2, col2] = 0
                    row2, col2 = next_capture[2], next_capture[3]
                    self.board[row2, col2] = current_piece
                    self.capture_piece(next_capture)
                    reward += 2
                    
                    # Check for promotion
                    if abs(current_piece) == 1 and ((player == 1 and row2 == 5) or (player == -1 and row2 == 0)):
                        self.board[row2, col2] = 2 * player
                        current_piece = 2 * player
            
            winner = self.game_winner(self.board)
            if winner == player:
                reward += 12
            elif winner == -player:
                reward -= 12
            elif winner == 0:
                reward -= 1
        else:
            reward -= 3  # Invalid action penalty

        self.count += reward
        self.player = -player
        return [self.board, reward]
    
    def render(self):
        """Display the current board state with pieces and kings"""
        print("  " + " ".join(map(str, range(6))))
        for i, row in enumerate(self.board):
            print(f"{i} ", end="")
            for piece in row:
                if piece == 1:
                    print("P ", end="")  # Player 1 piece
                elif piece == -1:
                    print("E ", end="")  # Player -1 piece
                elif piece == 2:
                    print("K ", end="")  # Player 1 king
                elif piece == -2:
                    print("Q ", end="")  # Player -1 king
                else:
                    print(". ", end="")  # Empty space
            print()  # New line after each row

