learning_rate = 0.05
epsilon = 0.8
epsilon_decay = 0.998
discount_factor = 0.95
epochs = 5000
reward_values = {
    "capture": 2,
    "multi_captures": 7,
    "win": 18,
    "invalid": -3,
    "draw": -10,
    "repeated_state": -2,
    "promote": 4
}