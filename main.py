import checkers_env
from LearningAgent import LearningAgent
import matplotlib.pyplot as plt
import numpy as np

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
    agent = LearningAgent(step_size=0.1, epsilon=0.1, env=env)
    
    # Training
    print("Starting training...")
    rewards = agent.learning(episodes=1000)
    
    # Plot training progress
    plot_rewards(rewards)
    
    # Play a test game after training
    print("\nPlaying test game with trained agent...")
    play_game(env, agent)

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
        new_state, reward = env.step(action, env.player)
        total_reward += reward
        
        print(f"\nMove made: {action}")
        env.render()
        
        # Check if game is over
        if env.game_winner(new_state) is not None:
            done = True
            print(f"\nGame Over! Total reward: {total_reward}")
            winner = env.game_winner(new_state)
            if winner == 1:
                print("Player 1 wins!")
            elif winner == -1:
                print("Player -1 wins!")
            else:
                print("Draw!")

if __name__ == "__main__":
    main()