import numpy as np

def epsilon_greedy_policy(q_function, state, action_size, epsilon):
    if np.random.rand() < epsilon:
        # Explore: Select a random action
        return np.random.randint(action_size)
    else:
        # Exploit: Select the best action based on Q-values
        q_values = [q_function.predict(state, action) for action in range(action_size)]
        return np.argmax(q_values)
    
    