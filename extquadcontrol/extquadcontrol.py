import numpy as np
import numbers
import warnings


class ExtendedQuadratic():
    """
    A python object that represents the extended quadratic function
    f(x) = (1/2) x.T P x + q.T x + (1/2) x
    +
    {
        0      if Fx+g=0
        +infty otherwise
    }
    """

    def __init__(self, P, q, r, F=None, g=None):
        """
        Initialize an extended quadratic function by supplying:
            - P: (n,n) numpy array
            - q: (n,) numpy array
            - r: number
            - F: (p,n) numpy array
            - g: (p,) numpy array
        """
        self.P = P
        self.q = q.flatten()
        self.r = r

        # check shapes of P,q,r
        assert len(self.P.shape) == 2, "P has the wrong dimensions"
        assert len(self.q.shape) == 1, "q has the wrong dimensions"
        n, m = self.P.shape
        assert n == m, "P is not square"
        assert self.q.shape[0] == n, "q is the wrong shape"
        assert isinstance(self.r, numbers.Real), "r must be number"

        # check shapes of F,g
        if F is not None and g is not None:
            assert len(F.shape) == 2, "F has the wrong dimensions"
            assert len(g.shape) == 1, "g has the wrong dimensions"
            assert F.shape[1] == n, "F is wrong shape"
            p = F.shape[0]
            assert g is not None and g.shape[0] == p, "g is wrong shape"
            self.F = F
            self.g = g
        else:
            self.F = np.empty((0, n))
            self.g = np.empty(0)

    def __repr__(self):
        """
        Evaluates the extended quadratic.
        """
        coefs = '\nP: ' + str(self.P) + '\n' + 'q: ' + \
            str(self.q) + '\n' + 'r: ' + str(self.r)
        eq = '\n F: ' + str(self.F) + '\ng: ' + str(self.g)
        return coefs + '\n' + eq

    @property
    def n(self):
        return self.P.shape[0]

    @property
    def p(self):
        return self.F.shape[0]

    @property
    def convex(self):
        if self.n == 0:
            return True
        _, _, V2 = self.reduced_form()
        return np.all(
            np.greater_equal(
                np.linalg.eigvals(
                    V2.T@self.P@V2
                ), -1e-8
            )
        )

    def __call__(self, x=None):
        n = self.n
        if n == 0:
            return .5 * self.r
        else:
            assert x is not None, "Must supply argument"
            assert len(x.shape) == 1, "x has wrong dimensions"
            assert x.shape[0] == n, "x has wrong shape"

        if self.p == 0:
            satisfies_equality_constraints = True
        else:
            satisfies_equality_constraints = np.allclose(
                np.dot(self.F, x) + self.g, 0)

        if not satisfies_equality_constraints:
            return float("inf")
        else:
            return .5 * x@self.P@x + self.q@x + .5 * self.r

    def reduced_form(self):
        """
        Returns:
            -x0: particular solution to Fx+g=0
            -V1: first part of SVD
            -V2: second part of SVD
        """
        if self.p == 0:
            return np.zeros(self.n), np.empty((self.n, 0)), np.eye(self.n)

        U, S, Vt = np.linalg.svd(self.F, full_matrices=True)
        rank = np.linalg.matrix_rank(self.F)

        U1 = U[:, :rank]
        V1 = Vt[:rank].T
        V2 = Vt[rank:].T
        Sigma = S[:rank]

        x0 = -V1@np.diag(1. / Sigma)@U1.T@self.g

        if not np.allclose(self.F@x0 + self.g, 0):
            warnings.warn("Not proper")

        self.F = V1.T
        self.g = np.diag(1. / S[:rank]) @ U1.T @ self.g

        return x0, V1, V2

    def __add__(f, g):
        """
        h(x) = f(x) + g(x)
        """
        if isinstance(g, numbers.Real):
            h = ExtendedQuadratic(
                f.P + g,
                f.q + g,
                f.r + g,
                f.F,
                f.g
            )
        else:
            Fnew = np.r_[f.F, g.F]
            gnew = np.r_[f.g, g.g]
            h = ExtendedQuadratic(
                f.P + g.P,
                f.q + g.q,
                f.r + g.r,
                Fnew,
                gnew
            )
            h.reduced_form()
        return h

    def __mul__(f, a):
        """
        h(x) = a*f(x)
        """
        h = ExtendedQuadratic(a * f.P, a * f.q, a * f.r, f.F, f.g)

        return h

    def __div__(f, a):
        """
        h(x) = f(x)/a
        """
        return (1. / a) * f

    def __truediv__(f, a):
        """
        h(x) = f(x)/a
        """
        return (1. / a) * f

    def __rmul__(f, a):
        return f.__mul__(a)

    def __rdiv__(f, a):
        return f.__div__(a)

    def __eq__(f, g):
        return f.distance(g) <= 1e-8

    def affine_composition(f, A, b, reduced_form=True):
        """
        h(x) = f(Ax+b).
        """
        h = ExtendedQuadratic(
            A.T@f.P@A,
            A.T@f.P@b + A.T@f.q,
            b@f.P@b + 2 * f.q@b + f.r,
            f.F@A,
            f.F@b + f.g
        )

        if h.p > 0 and reduced_form:
            h.reduced_form()

        return h

    def distance(f, g):
        # d(f,g)
        if not f.equality_constraints_equal(g):
            return float("inf")

        x0, _, V2 = f.reduced_form()

        metric =  np.linalg.norm(V2.T@(f.P - g.P)@V2, ord='fro')**2 + \
            2 * np.linalg.norm(V2.T@(f.P@x0 + f.q - g.P@x0 - g.q), ord=2)**2 + \
            (x0.T @ (f.P@x0 + 2 * f.q - g.P@x0 - 2 * g.q) + f.r - g.r)**2

        return metric

    def equality_constraints_equal(f, g):
        x0, V1, V2 = f.reduced_form()
        x0_tilde, V1_tilde, V2_tilde = g.reduced_form()

        c1 = np.allclose(V1_tilde.T@V2, 0)
        c2 = np.allclose(V1.T@V2_tilde, 0)
        c3 = np.allclose(f.F@x0_tilde + f.g, 0)
        c4 = np.allclose(g.F@x0 + g.g, 0)

        return c1 and c2 and c3 and c4

    def convex_indices(self, indices):
        assert min(indices) >= 0 and max(indices) < self.n, "Invalid indices"

        _, _, V2 = self.reduced_form()
        u_mask = np.zeros(self.n, np.bool)
        u_mask[indices] = True
        P_uu = np.atleast_2d(self.P[u_mask, :][:, u_mask])
        V2 = V2[u_mask, :]

        if ((V2.T@P_uu@V2).shape == (0, 0)):
            return True, True
        min_eigval = np.min(np.linalg.eigvals(V2.T@P_uu@V2))

        strictly_convex = min_eigval > 0
        convex = min_eigval >= -1e-8

        return convex, strictly_convex

    def partial_minimization(self, indices):
        """
        Optimal value of optimization problem
            minimize_u f(x,u)
        is a convex quadratic. Returns the new quadratic and (A,b) where u=Ax+b.
        indices is a subset of {1,...,n+m} to minimize
        """
        convex, strictly_convex = self.convex_indices(indices)
        assert convex, "not extended quadratic because not convex"

        n_u = len(indices)
        n_x = self.n - n_u

        u_mask = np.zeros(self.n, np.bool)
        u_mask[indices] = True
        x_mask = ~u_mask

        q_u = self.q[u_mask]
        P_ux = np.atleast_2d(self.P[u_mask, :][:, x_mask])
        P_uu = np.atleast_2d(self.P[u_mask, :][:, u_mask])
        g = self.g
        F_x = self.F[:, x_mask]
        F_u = self.F[:, u_mask]
        F_u_pinv = np.linalg.pinv(F_u)

        KKT_matrix = np.r_[
            np.c_[P_uu, F_u.T],
            np.c_[F_u, np.zeros((self.p, self.p))]
        ]
        KKT_matrix_pinv = np.linalg.pinv(KKT_matrix)

        Ft = (np.eye(F_u.shape[0]) - F_u@F_u_pinv)@F_x
        gt = (np.eye(F_u.shape[0]) - F_u@F_u_pinv)@g

        Ap = np.r_[P_ux, F_x]
        bp = np.r_[q_u, g]

        if n_x > 0 and not strictly_convex:
            temp = ExtendedQuadratic(
                np.zeros((n_x, n_x)), np.zeros(n_x), 0, Ft, gt)
            x_0, V1, V2 = temp.reduced_form()

            Rhs = np.c_[
                Ap@V2, Ap@x_0 + bp
            ]
            assert np.allclose(
                (np.eye(KKT_matrix.shape[0]) - KKT_matrix@KKT_matrix_pinv)@Rhs,
                0
            ), "not extended quadratic because range constraint does not hold"

        A = np.zeros((self.n, n_x))
        A[x_mask, :] = np.eye(n_x)

        b = np.zeros(self.n)
        b[x_mask] = np.zeros(n_x)

        res = -np.c_[np.eye(n_u), np.zeros((n_u, self.p))] @ \
            KKT_matrix_pinv @ \
            np.c_[Ap, bp]
        A[u_mask, :] = res[:, :-1]
        b[u_mask] = res[:, -1]

        f = ExtendedQuadratic(self.P, self.q, self.r)
        f = f.affine_composition(A, b)
        f.F = Ft
        f.g = gt

        return f, A[u_mask, :], b[u_mask]


def dp_infinite(sample, num_iterations, N, gamma=1):
    """
    Arguments:
        - sample(N): function that gives a batch sample of
            - A_t: (N,K,n,n) numpy array
            - B_t: (N,K,n,m) numpy array
            - c_t: (N,K,n) numpy array
            - g_t: length-N list of length-K list of ExtendedQuadratics
            - Pi_t: (K,K) numpy array
        - T: horizon length
        - N: number of monte carlo iterations
    This function performs the dynamic programming recursion described in the paper [].
    It returns an length T+1 list of length-K list of ExtendedQuadratics representing the cost-to-go functions.
    It also returns a length T list of length-K list of ExtendedQuadratics represneting the state-action cost-to-go functions.
    It also returns a length T list of length-K list of policies, where each policy is a matrix+vector representing an affine function.
        e.g. Vs[t][s] or Qs[t][s] or policies[t][s]
    """
    # initialize the cost-to-go functions and policies
    A, B, c, g, Pi = sample(1)
    _, K, n, _ = A.shape
    g_T = [ExtendedQuadratic(np.zeros((n, n)), np.zeros(n), 0)
           for _ in range(K)]

    def sample_time_invariant(t, N):
        A, B, c, g, Pi = sample(N)
        return A, B, c, (gamma**t) * g, Pi
    Vs, Qs, policies = dp_finite(sample_time_invariant, g_T, num_iterations, N)

    return Vs[0], Qs[0], policies[0]


def dp_finite(sample, g_T, T, N):
    """
    Arguments:
        - sample(t): function that gives a batch sample of
            - A_t: (N,K,n,n) numpy array
            - B_t: (N,K,n,m) numpy array
            - c_t: (N,K,n) numpy array
            - g_t: length-N list of length-K list of ExtendedQuadratics
            - Pi_t: (K,K) numpy array
        - g_T: list of length-K list of ExtendedQuadratics
        - T: horizon length
        - N: number of monte carlo iterations
    This function performs the dynamic programming recursion described in the paper [].
    It returns an length T+1 list of length-K list of ExtendedQuadratics representing the cost-to-go functions.
    It also returns a length T list of length-K list of ExtendedQuadratics represneting the state-action cost-to-go functions.
    It also returns a length T list of length-K list of policies, where each policy is a matrix+vector representing an affine function.
        e.g. Vs[t][s] or Qs[t][s] or policies[t][s]
    """
    # initialize the cost-to-go functions and policies
    Vs = [[] for _ in range(T + 1)]
    Qs = [[] for _ in range(T)]
    policies = [[] for _ in range(T)]
    Vs[-1] = g_T

    # backward recursion
    for t in range(T)[::-1]:
        Qs[t], n, m, K = get_qs(sample, Vs[t + 1], N, t)
        for Q in Qs[t]:
            V, policy_A, policy_b = Q.partial_minimization(np.arange(n, n + m))
            Vs[t].append(V)
            policies[t].append((policy_A, policy_b))
    return Vs, Qs, policies


def get_qs(sample, V, N, t):
    Qs = []
    A, B, c, g, Pi = sample(t, N)
    _, _, _, m = B.shape
    _, K, n, _ = A.shape
    for s in range(K):
        Q = ExtendedQuadratic(np.zeros((n + m, n + m)), np.zeros(n + m), 0)
        for k in range(N):
            Q += g[k][s] / N
            for sprime in range(K):
                Q += Pi[sprime, s] / N * \
                    V[sprime].affine_composition(
                        np.c_[A[k][s], B[k][s]], c[k][s])
        Qs.append(Q)
    return Qs, n, m, K


def dp_finite_mpi(sample, g_T, T, N, comm):
    """
    Arguments:
        - sample(t): function that gives a batch sample of
            - A_t: (N,K,n,n) numpy array
            - B_t: (N,K,n,m) numpy array
            - c_t: (N,K,n) numpy array
            - g_t: length-N list of length-K list of ExtendedQuadratics
            - Pi_t: (K,K) numpy array
        - g_T: list of length-K list of ExtendedQuadratics
        - T: horizon length
        - N: number of monte carlo iterations
    This function performs the dynamic programming recursion described in the paper [].
    It returns an length T+1 list of length-K list of ExtendedQuadratics representing the cost-to-go functions.
    It also returns a length T list of length-K list of ExtendedQuadratics represneting the state-action cost-to-go functions.
    It also returns a length T list of length-K list of policies, where each policy is a matrix+vector representing an affine function.
        e.g. Vs[t][s] or Qs[t][s] or policies[t][s]
    """
    # initialize the cost-to-go functions and policies

    nprocs = comm.Get_size()
    myrank = comm.Get_rank()

    N_per_proc = int(N // nprocs) + 1

    if myrank == 0:
        Vs = [[] for _ in range(T + 1)]
        Qs = [[] for _ in range(T)]
        policies = [[] for _ in range(T)]
        Vs[-1] = g_T

    # backward recursion
    for t in range(T)[::-1]:
        if myrank == 0:
            data = [(N_per_proc, Vs[t + 1])] * (nprocs)
        else:
            data = None
        N, V = comm.scatter(data, root=0)
        Qs_scattered, n, m, K = get_qs(sample, V, N, t)
        data = comm.gather(Qs_scattered, root=0)
        if myrank == 0:
            for s in range(K):
                Q = ExtendedQuadratic(
                    np.zeros((n + m, n + m)), np.zeros(n + m), 0)
                for d in data:
                    Q += d[s] / nprocs
                Qs[t].append(Q)
                V, policy_A, policy_b = Q.partial_minimization(
                    np.arange(n, n + m))
                Vs[t].append(V)
                policies[t].append((policy_A, policy_b))
    if myrank == 0:
        return Vs, Qs, policies
    else:
        return None
