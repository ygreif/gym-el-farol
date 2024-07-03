import random
import numpy as np


class EGreedyAgent(object):
    def __init__(self, action_space, **userconfig):
        self.action_space = action_space
        self.config = {
            "learning_rate": 0.1,  # Reward multiplier
            "initial_epsilon": 1,  # Exploration probability
            "epsilon_decay": 0.01,  # Exploration reduction over time
            "final_epsilon": 0.1  # Final exploration probability
        }
        self.config.update(userconfig)
        self.q = [0, 0]
        self.epsilon = self.config["initial_epsilon"]

    def act(self):
        a = None
        if random.random() < self.epsilon or np.count_nonzero(self.q) == 0:
            a = np.random.randint(0, 2)
        else:
            a = np.argmax(self.q)
        self.prev_action = a
        return a

    def learn(self, reward):
        self.q[self.prev_action] += reward * self.config["learning_rate"]

    def decay_epsilon(self):
        self.epsilon = max(self.config["final_epsilon"], self.epsilon - self.config["epsilon_decay"])
