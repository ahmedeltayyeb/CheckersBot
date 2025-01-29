import random
import numpy as np

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def select_action(self, state):
        """
        Select a random valid action based on the given state.

        Args:
            state (np.array): The current board state.

        Returns:
            action (list) or None: A random valid move, or None if no valid moves exist.
        """
        # Calculate valid moves based on the provided state
        valid_moves = self.env.valid_moves(self.env.player) if np.array_equal(self.env.board, state) else []
        if not valid_moves:
            return None  # No valid moves available
        return random.choice(valid_moves)  # Randomly choose a valid move
    
    def play_vs_random(self, env, trained_agent, epochs):
        """
        Test the trained agent against a random agent.

        Args:
            env: The checkers environment.
            trained_agent: The trained learning agent.
            epochs: Number of matches to play.

        Returns:
            dict: Match statistics including wins, losses, and draws.
        """
        random_agent = RandomAgent(env)
        results = {"trained_agent_wins": 0, "random_agent_wins": 0}

        for episode in range(epochs):
            env.reset()
            done = False
            while not done:
                # Get the current state of the board
                current_state = env.board.copy()

                # Trained agent's turn
                if env.player == 1:
                    action = trained_agent.select_action(current_state)  # Pass the current state
                    if action is None:
                        # No valid moves for the trained agent, random agent wins
                        results["random_agent_wins"] += 1
                        break
                    _, _, done, info = env.step(action, env.player, trained_agent)

                # Random agent's turn
                else:
                    action = random_agent.select_action(current_state)  # Pass the current state
                    if action is None:
                        # No valid moves for the random agent, trained agent wins
                        results["trained_agent_wins"] += 1
                        break
                    _, _, done, info = env.step(action, env.player, random_agent)

                # Check for game end
                if done:
                    if info.get("winner") == 1:
                        results["trained_agent_wins"] += 1
                    elif info.get("winner") == -1:
                        results["random_agent_wins"] += 1
                    break

            # print(f"Episode {episode + 1}/{epochs}: Winner {info.get('winner', 'None')}")

        return results
    

def play_against_agent(env, agent):
    """Let the trained agent play against itself"""
    env.reset()
    done = False
    player_1_reward = 0  # Reward for player 1
    player_2_reward = 0  # Reward for player -1
    
    print("\nStarting new game:")
    env.render()
    print("Final result of test game:")

    
    while not done:
        current_state = env.board.copy()
        action = agent.select_action(current_state)
        env.render()
        
        if action is None:
            print("No valid moves available")
            break
        
        # Make move
        _, reward, done, info = env.step(action, env.player, agent)
        
        # Accumulate rewards for each player
        if env.player == 1:
            player_1_reward += reward
        else:
            player_2_reward += reward
        
    
    if done:
        print("\nGame Over!")
        print(f"\nTotal reward for Player 1: {player_1_reward}")
        print(f"Total reward for Player -1: {player_2_reward}")   
    if info.get('winner') == 1:
        print("Player 1 wins!")
    elif info.get('winner') == -1:
        print("Player -1 wins!")
    else:
        print("Draw!")


def play_vs_human(env, agent):
    """
    Play a game against the trained agent.
    
    Args:
        env: The Checkers environment.
        agent: The trained agent.
    """
    env.reset()
    done = False
    human_player = 1  # Assume human is Player 1 (you can switch to -1 for Player -1)
    
    print("\nGame Start! You are Player 1 (P)")
    env.render()

    while not done:
        if env.player == human_player:
            # Human turn
            print("\nYour Turn!")
            valid_moves = env.valid_moves(env.player)
            if not valid_moves:
                print("No valid moves available! You lose!")
                break

            print(f"Valid Moves: {valid_moves}")
            move = None
            while move not in valid_moves:
                try:
                    move = input("Enter your move in the format 'x1 y1 x2 y2' (e.g., '2 3 3 4'): ")
                    move = list(map(int, move.split()))
                except ValueError:
                    print("Invalid input format. Try again.")

                if move not in valid_moves:
                    print("Invalid move. Try again.")

            _, _, done, info = env.step(move, env.player, agent)
            env.render()
            if done:
                print("\nGame Over!")
                winner = info.get('winner')
                if winner == human_player:
                    print("You win!")
                elif winner == -human_player:
                    print("Agent wins!")
                else:
                    print("It's a draw!")
        else:
            # Agent turn
            print("\nAgent's Turn...")
            valid_moves = env.valid_moves(env.player)
            if not valid_moves:
                print("No valid moves available for the agent! You win!")
                break

            action = agent.select_action(env.board)
            print(f"Agent chose move: {action}")
            _, _, done, info = env.step(action, env.player, agent)
            env.render()
            if done:
                print("\nGame Over!")
                winner = info.get('winner')
                if winner == human_player:
                    print("You win!")
                elif winner == -human_player:
                    print("Agent wins!")
                else:
                    print("It's a draw!")

