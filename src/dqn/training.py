import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .agent import DQNAgent
from ..environment.checkers_env import CheckersEnv

def train_dqn(episodes=10000, target_update=10, eval_interval=100, save_path='models/dqn_checkers.pt'):
    env = CheckersEnv()
    agent = DQNAgent()
    
    rewards = []
    avg_rewards = []
    
    for episode in tqdm(range(episodes)):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                done = True
                continue
                
            action = agent.choose_action(state, valid_moves)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            episode_reward += reward
            
        # Update target network periodically
        if episode % target_update == 0:
            agent.update_target_network()
            
        # Decay exploration rate
        agent.update_epsilon()
        
        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-100:])
        avg_rewards.append(avg_reward)
        
        if (episode + 1) % eval_interval == 0:
            print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            
            # Save model periodically
            agent.save_model(save_path)
    
    # Final save
    agent.save_model(save_path)
    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.title('DQN Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(['Episode Reward', 'Average Reward (100 ep)'])
    plt.savefig('dqn_training_progress.png')
    plt.show()
    
    return agent