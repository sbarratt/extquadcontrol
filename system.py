import numpy as np

class InfiniteHorizonSystem():
    def __init__(self, sample, K):
        self.sample = sample
        self.K = K

    def simulate(self, x0, s0, T, policy, gamma=1.):
        cost = 0.
        X = []
        U = []
        Modes = []
        x, s = x0, s0

        for t in range(T):
            u = policy(t,x,s)
            xu = np.r_[x,u]

            X.append(x)
            U.append(u)
            Modes.append(s)

            A,B,c,g,Pi = self.sample(1)
            A = A[0,s]
            B = B[0,s]
            c = c[0,s]
            g = g[0][s]
            cost += gamma**t * g(xu) / T
            s = np.random.choice(np.arange(self.K),p=Pi[:,s])
            x = A@x + B@u + c
        X.append(x)
        Modes.append(s)

        return np.array(X), np.array(U), Modes, cost

class FiniteHorizonSystem():
    def __init__(self, sample, g_T, K):
        self.sample = sample
        self.K = K
        self.g_T = g_T

    def simulate(self, x0, s0, T, policy, t0):
        cost = 0.
        X = []
        U = []
        Modes = []
        x, s = x0, s0

        for t in range(t0,T):
            u = policy(t,x,s)
            xu = np.r_[x,u]

            X.append(x)
            U.append(u)
            Modes.append(s)
            A,B,c,g,Pi = self.sample(t,1)
            A = A[0,s]
            B = B[0,s]
            c = c[0,s]
            g = g[0][s]
            cost += g(xu)
            s = np.random.choice(np.arange(self.K),p=Pi[:,s])
            x = A@x + B@u + c
        cost += self.g_T[s](x)
        X.append(x)
        Modes.append(s)

        return np.array(X), np.array(U), Modes, cost
