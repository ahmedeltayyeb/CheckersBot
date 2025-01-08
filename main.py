import checkers_env
from LearningAgent import LearningAgent
import matplotlib.pyplot as plt
import os
from config import learning_rate, epsilon, discount_factor, episodes
from play_games import play_vs_human, play_against_agent, RandomAgent



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
    # play_against_agent(env, agent)

    test_games = RandomAgent(env=env)

    # Test against random agent
    results = test_games.play_vs_random(env, agent, episodes=50)

    # Print the results
    print(f"Results after 50 games:")
    print(f"Trained Agent Wins: {results['trained_agent_wins']}")
    print(f"Random Agent Wins: {results['random_agent_wins']}")
    print(f"Draws: {results['draws']}")


    # play_vs_human(env, agent)


if __name__ == "__main__":
    main()