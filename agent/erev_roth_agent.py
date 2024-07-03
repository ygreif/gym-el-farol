import random
from collections import defaultdict


class ErevRothAgent(object):
    def __init__(self, action_space, **userconfig):
        self.action_space = action_space
        self.config = {
            "init_mean": 1.0,  # Initialize Q values with this mean
            "init_std": 0.0,  # Initialize Q values with this standard deviation
            "learning_rate": 0.5,  # Reward multiplier
            "forget_rate": 0.9975  # Increase reaction speed to dynamic environment
        }
        self.config.update(userconfig)
        self.q = defaultdict(lambda: random.normalvariate(self.config["init_mean"], self.config["init_std"]))

    def act(self):
        total = sum([self.q[a] for a in range(0, self.action_space.n)])
        r = random.random()
        cum = 0
        for a, p in self.q.items():
            cum += p / total
            if r < cum:
                self.prev_action = a
                return a
        raise Exception("No value selected")

    def learn(self, reward):
        self.q[self.prev_action] += reward * self.config["learning_rate"]
        if self.config["forget_rate"] != 1:
            for key in self.q:
                self.q[key] *= self.config["forget_rate"]
