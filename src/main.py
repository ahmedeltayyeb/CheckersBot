import checkers_env
from LearningAgent import LearningAgent
import matplotlib.pyplot as plt
import os
from config import learning_rate, epsilon, discount_factor, epochs
from play_games import play_vs_human, play_against_agent, RandomAgent


def plot_rewards(rewards):
    """Plot the rewards over epochs"""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


def main():
    # Initialize environment and pre-trained agent
    env = checkers_env.checkers_env()
    pre_trained_agent = LearningAgent(learning_rate=learning_rate, epsilon=epsilon, discount_factor=discount_factor, env=env)
    
    # Load Q-table for the pre-trained agent
    pretrained_q_table_file = "src/q-tables/pre_trained_qtable.pkl"
    if os.path.exists(pretrained_q_table_file):
        pre_trained_agent.q_learning.load_qtable(pretrained_q_table_file)
        print("Loaded Pre-trained Q-table")
    
    agent = LearningAgent(learning_rate=learning_rate, epsilon=epsilon, discount_factor=discount_factor, env=env)
    random_agent = RandomAgent(env=env)
    print("Starting training...")

    # Load existing Q-table if it exists
    current_training = "src/q-tables/new_q_table.pkl"
    if os.path.exists(current_training):
        agent.q_learning.load_qtable(current_training)
        print("Loaded Latest Q-table")

    rewards = agent.train(
        epochs=epochs,
        pretrained_agent=pre_trained_agent,
        random_agent=random_agent,
        random_agent_percentage=0.3,
        self_play_percentage=0.3
        )
    
    # Save Q-table for next run
    agent.q_learning.save_qtable(current_training)
    print("Training completed! Saved Q-table")
    
    # Plot training progress
    plot_rewards(rewards)
    
    # Play a test game after training to show the agent's behavior
    play_against_agent(env, agent)

    # Test against random agent with as many games as specified (epochs)
    results = random_agent.play_vs_random(env, agent, epochs=1000)

    # Print the results
    print(f"Results after 1000 games:")
    print(f"Trained Agent Wins: {results['trained_agent_wins']}")
    print(f"Random Agent Wins: {results['random_agent_wins']}")

    
    # A human can play against the trained agent
    play_vs_human(env, agent)


# def main():
#     # Initialize environment and pre-trained agent
#     env = checkers_env.checkers_env()

    
#     agent = LearningAgent(learning_rate=learning_rate, epsilon=epsilon, discount_factor=discount_factor, env=env)

#     # Load existing Q-table if it exists
#     current_training = "pre_trained_qtable.pkl"


#     rewards = agent.train(
#         epochs=epochs,
#         random_agent_percentage=0,
#         self_play_percentage=1
#         )
    
#     # Save Q-table for next run
#     agent.q_learning.save_qtable(current_training)
#     print("Training completed! Saved Q-table")
    
#     # Plot training progress
#     plot_rewards(rewards)
    
#     # Play a test game after training to show the agent's behavior
#     play_against_agent(env, agent)
#     random_agent = RandomAgent(env=env)

#     # Test against random agent with as many games as specified (epochs)
#     results = random_agent.play_vs_random(env, agent, epochs=1000)

#     # Print the results
#     print(f"Results after 1000 games:")
#     print(f"Trained Agent Wins: {results['trained_agent_wins']}")
#     print(f"Random Agent Wins: {results['random_agent_wins']}")
#     print(f"Draws: {results['draws']}")
    
#     # A human can play against the trained agent
#     play_vs_human(env, agent)


if __name__ == "__main__":
    main()