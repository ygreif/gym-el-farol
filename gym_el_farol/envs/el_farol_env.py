from __future__ import print_function

from gym import Env
from gym.spaces import Discrete

class ElFarolEnv(Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, n_agents=100, threshold=60, g=10, s=5, b=1):
        if g < s or s < b:
            raise Exception("rewards must be ordered g > s > b")

        self.n_agents = n_agents
        self.action_space = Discrete(2)
        # observe 0 if did not attend, otherwise observe number of agents who atteneded
        self.observation_space = Discrete(n_agents)
        self.reward_range = (b, g)
        def reward_func(action, n_attended):
            if action == 0:
                return s
            elif n_attended <= threshold:
                return g
            else:
                return b
        self.reward_func = reward_func
        self.prev_action = [self.action_space.sample() for _ in range(n_agents)]

    def _step(self, action):
        n_attended = sum(action)
        observation = [n_attended if a else 0 for a in action]
        reward = [self.reward_func(a, n_attended) for a in action]

        self.prev_action = action
        return observation, reward, False, ()

    def _reset(self):
        pass

    def _render(self, mode='human', close=False):
        if mode == 'human':
            print(str(sum(self.prev_action)))
