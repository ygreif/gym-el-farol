from __future__ import print_function

from gymnasium import Env
from gymnasium.spaces import Discrete


class ElFarolEnv(Env):
    def __init__(self, n_agents=100, threshold=60, g=10, sg=5, sb=1, b=1):
        if g < sg or sg < sb or sb < b:
            raise Exception("rewards must be ordered g > sg > sb > b")

        self.n_agents = n_agents
        self.action_space = Discrete(2)
        self.observation_space = Discrete(n_agents)
        self.reward_range = (b, g)
        self.threshold = threshold
        self.sg = sg
        self.sb = sb
        self.g = g
        self.b = b
        self.prev_action = [self.action_space.sample() for _ in range(n_agents)]
        self.iteration = 0

    def modify_threshold(self, change):
        self.threshold = int(self.threshold + self.threshold * change)

    def reward_func(self, action, n_attended):
        if action == 0:
            if n_attended >= self.threshold:
                return self.sg
            if n_attended < self.threshold:
                return self.sb
        elif n_attended <= self.threshold:
            return self.g
        else:
            return self.b

    def step(self, action):
        self.iteration += 1
        n_attended = sum(action)
        print(str(self.iteration) + ", " + str(n_attended) + ", " + str(self.threshold))
        reward = [self.reward_func(a, n_attended) for a in action]
        self.prev_action = action
        return n_attended, reward, False, ()
