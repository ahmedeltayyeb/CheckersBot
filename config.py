learning_rate = 0.1
epsilon = 0.8
epsilon_decay = 0.995
discount_factor = 0.9
episodes = 10000
reward_values = {
    "capture": 3,
    "multi_captures": 7,
    "win": 20,
    "invalid": -5,
    "draw": -10,
    "repeated_state": -3,
    "promote": 4
}