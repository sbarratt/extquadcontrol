class Policy():
    def __init__(self):
        pass

    def __call__(self, t, x, s):
        raise NotImplementedError

class AffinePolicy(Policy):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, t, x, s):
        K, k = self.policies[t][s]
        return K@x + k

class TimeInvariantAffinePolicy(Policy):
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, t, x, s):
        K, k = self.policy[s]
        return K@x + k