from collections import defaultdict

class FuzzyPureNash(object):
    def __init__(self, threshold=.95):
        self.action_counts_by_agent = defaultdict(lambda: defaultdict(lambda: int()))
        self.threshold = threshold

    def step(self, action):
        for agent, a in enumerate(action):
            self.action_counts_by_agent[agent][a] += 1

    def in_equilibria(self):
        for action_counts in self.action_counts_by_agent.values():
            if max(action_counts.values()) / float(sum(action_counts.values())) < self.threshold:
                return False
        return True
