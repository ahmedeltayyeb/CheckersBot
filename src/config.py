# BATCH_SIZE is the number of transitions sampled from the replay buffer
# tau is the update rate of the target network

learning_rate = 0.05
epsilon = 0.8
epsilon_decay = 0.998
discount_factor = 0.95
epochs = 500
tau = 0.005
reward_values = {
    "capture": 2,
    "multi_captures": 7,
    "win": 18,
    "invalid": -3,
    "lose": -10,
    "repeated_state": -2,
    "promote": 4
}