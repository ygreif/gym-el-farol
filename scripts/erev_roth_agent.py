import random
from collections import defaultdict

from gym.spaces import Discrete, Tuple
from gym_el_farol.envs import ElFarolEnv, FuzzyPureNash

class ErevRothAgent(object):
    def __init__(self, observation_space, action_space, **userconfig):
        if not isinstance(observation_space, Discrete):
            raise Exception('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        if not isinstance(action_space, Discrete):
            raise Exception('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n
        self.config = {
            "init_mean" : 1.0,      # Initialize Q values with this mean
            "init_std" : 0.0,       # Initialize Q values with this standard deviation
            "learning_rate" : 1.0}
        self.config.update(userconfig)
        self.q = defaultdict(lambda: random.normalvariate(self.config["init_mean"], self.config["init_std"]))

    def act(self):
        # replace with numpy
        total = sum([self.q[a] for a in range(0, self.action_space.n)])
        r = random.random()
        cum = 0
        for a, p in self.q.items():
            cum += p / total
            if r < cum:
                self.prev_action = a
                return a
        raise Exception("No value selected", p)

    def learn(self, reward):
        self.q[self.prev_action] += reward * self.config["learning_rate"]
        for key in self.q:
            self.q[key] *= .99

def iterate(agents, env):
    actions = [a.act() for a in agents]
    obs, reward, _, _ = env.step(actions)
    for agent, reward in zip(agents, reward):
        agent.learn(reward)
    return actions

def iterations_to_equilbira(agents, env):
    nash = FuzzyPureNash()
    for iter in range(0, 5000000):
        if iter % 50 == 0 and iter > 0:
            print iter
            if nash.in_equilibria():
                return iter
            nash = FuzzyPureNash()
        actions = iterate(agents, env)
        nash.step(actions)
    for agent in agents:
        print agent.q[0] / (agent.q[0] + agent.q[1])
    return False

n_agents = 100
env = ElFarolEnv(n_agents=n_agents, threshold=5)
agents = []
for i in range(0, n_agents):
    agents.append(ErevRothAgent(env.observation_space, env.action_space))
print iterations_to_equilbira(agents, env)
