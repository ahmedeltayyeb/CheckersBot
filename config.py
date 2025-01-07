learning_rate = 0.2
epsilon = 0.5
epsilon_decay = 0.998
discount_factor = 0.95
episodes = 1000
reward_values = {
    "capture": 3,
    "multi_captures": 5,
    "win": 15,
    "lose": -15,
    "invalid": -3,
    "draw": -3,
    "repeated_state": -1,
    "promote": 3
}