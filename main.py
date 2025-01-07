import checkers_env
from LearningAgent import LearningAgent
import matplotlib.pyplot as plt
import os
from config import learning_rate, epsilon, discount_factor, episodes


def plot_rewards(rewards):
    """Plot the rewards over episodes"""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

def main():
    # Initialize environment and agent
    env = checkers_env.checkers_env()
    agent = LearningAgent(learning_rate=learning_rate, epsilon=epsilon, discount_factor=discount_factor, env=env)
    
    # Load existing Q-table if it exists
    q_table_file = "qtable.pkl"
    if os.path.exists(q_table_file):
        agent.q_learning.load_qtable(q_table_file)
        print("Loaded Q-table")
    
    # Training
    print("Starting training...")
    rewards = agent.train(episodes=episodes)
    
    # Save Q-table for next run
    agent.q_learning.save_qtable(q_table_file)
    print("Saved Q-table")
    
    # Plot training progress
    plot_rewards(rewards)
    
    # Play a test game after training
    print("\nPlaying test game with trained agent...")
    play_game(env, agent)
    
    play_against_agent(env, agent)

    
def play_game(env, agent):
    """Play a single game with the trained agent"""
    env.reset()
    done = False
    total_reward = 0
    
    print("\nStarting new game:")
    env.render()
    
    while not done:
        current_state = env.board.copy()
        action = agent.select_action(current_state)
        
        if action is None:
            print("No valid moves available")
            break
        
        # Make move
        _, reward, done, info = env.step(action, env.player, agent)
        total_reward += reward
        
        print(f"\nMove made: {action}")
        env.render()
        
        if done:
            print(f"\nGame Over! Total reward: {total_reward}")
            if info.get('winner') == 1:
                print("Player 1 wins!")
            elif info.get('winner') == -1:
                print("Player -1 wins!")
            else:
                print("Draw!")

def play_against_agent(env, agent):
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


if __name__ == "__main__":
    main()