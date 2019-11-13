import unittest
import numpy as np
import cvxpy as cp
from extquadcontrol import ExtendedQuadratic, dp_infinite


def extended_quadratic_to_cvx(f, x):
    if f.F.shape[0] == 0:
        return .5 * cp.quad_form(x, f.P) + f.q * x + .5 * f.r, []
    else:
        return .5 * cp.quad_form(x, f.P) + f.q * x + .5 * f.r, [f.F * x + f.g == 0]


class TestExtQuadControl(unittest.TestCase):

    def test_extended_quadratics(self):
        f = ExtendedQuadratic(np.array([[1]]), np.array([1]), 1)
        g = ExtendedQuadratic(np.array([[-1]]), np.array([2]), 2)
        h = ExtendedQuadratic(np.array(
            [[1, 0], [0, -1]]), np.array([0, 0]), 0, np.array([[0, 1]]), np.array([0]))
        i = ExtendedQuadratic(np.array([[1, 0], [0, -1]]), np.array([0, 0]), 0)
        empt = ExtendedQuadratic(np.empty((0, 0)), np.empty(0), 5)

        assert f.convex
        convex, strictly_convex = f.convex_indices([0])
        assert convex and strictly_convex
        assert not g.convex
        convex, strictly_convex = g.convex_indices([0])
        assert not convex and not strictly_convex

        assert h.convex
        assert not i.convex
        convex, strictly_convex = i.convex_indices([0])
        assert convex
        convex, strictly_convex = i.convex_indices([1])
        assert not convex
        assert (h + i).convex

        assert f.n == 1
        assert f.p == 0
        assert h.n == 2
        assert h.p == 1

        # test call
        assert f(np.array([1])) == .5 * 1 + 1 + .5
        assert g(np.array([0])) == .5 * 2
        assert empt() == .5 * 5
        assert h(np.array([1, 1])) == float("inf")
        assert h(np.array([1, 0])) == .5

        # test reduced_form
        for _ in range(100):
            rand = ExtendedQuadratic(np.random.randn(10, 10), np.random.randn(
                10), .5, np.random.randn(2, 10), np.random.randn(2))
            x0, V1, V2 = rand.reduced_form()
            assert np.allclose(rand.F@x0 + rand.g, 0)
            assert np.linalg.matrix_rank(np.c_[V1, V2]) == rand.n
            assert np.allclose(V1.T@V1, np.eye(V1.shape[1]))
            assert np.allclose(V2.T@V2, np.eye(V2.shape[1]))
            assert np.allclose(V1.T@V2, 0)

        # test add
        assert (f + g).P == np.array([[0]])
        assert (f + g).q == np.array([3])
        assert (f + g).r == 3
        # assert np.allclose((h+i).F,np.array([[0,-1]]))
        # assert np.allclose((h+i).g,0)

        # test mul
        assert (.5 * f).P == np.array([[.5]])
        assert (.5 * f).q == np.array([.5])
        assert (.5 * f).r == .5

        # test div
        assert (f / 2).P == np.array([[.5]])
        assert (f / 2).q == np.array([.5])
        assert (f / 2).r == .5

        # test affine_composition
        P = np.random.randn(10, 10)
        P = P@P.T
        q = np.random.randn(10)
        r = .5
        F = np.random.randn(2, 10)
        g = np.random.randn(2)
        A = np.random.randn(10, 5)
        b = np.random.randn(10)
        f = ExtendedQuadratic(P, q, r, F, g)
        z = f.affine_composition(A, b, reduced_form=False)
        assert np.allclose(z.P, A.T@P@A)
        assert np.allclose(z.q, A.T@P@b + A.T@q)
        assert np.allclose(z.r, b.T@P@b + 2 * b.T@q + r)

        # test distance and equality
        h = f + f
        hpr = 2 * f
        assert h == hpr
        assert h != f

        # test partial minimization on a random problem instance vs cvx
        P = np.random.randn(10, 10)
        P = P@P.T
        q = np.random.randn(10)
        r = .5
        F = np.random.randn(2, 10)
        g = np.random.randn(2)
        f = ExtendedQuadratic(P, q, r, F, g)
        f.reduced_form()
        fmin, A, b = f.partial_minimization(np.arange(10))

        x = cp.Variable(10)
        fcvx, constraints = extended_quadratic_to_cvx(f, x)
        prob = cp.Problem(cp.Minimize(fcvx), constraints)
        result = prob.solve()
        assert np.allclose(result, fmin())
        assert np.allclose(x.value.flatten(), b, rtol=1e-2, atol=1e-2)

        # test partial minimization by generating random x's, solving in cvx
        # and checking against solution
        for _ in range(10):
            P = np.random.randn(10, 10)
            P = P@P.T
            q = np.random.randn(10)
            r = .5
            F = np.random.randn(2, 10)
            g = np.random.randn(2)
            f = ExtendedQuadratic(P, q, r, F, g)
            f.reduced_form()

            fmin, A, b = f.partial_minimization(np.arange(5))

            for _ in range(10):
                x = np.random.randn(5)
                fadj = f.affine_composition(
                    np.r_[np.eye(5), np.zeros((5, 5))], np.r_[np.zeros(5), x])
                xcvx = cp.Variable(5)
                fcvx, constraints = extended_quadratic_to_cvx(fadj, xcvx)
                prob = cp.Problem(cp.Minimize(fcvx), constraints)
                result = prob.solve()
                assert np.allclose(result, fmin(x))
                assert np.allclose(A@x + b, xcvx.value.flatten(), rtol=1e-2, atol=1e-2)

        # test partial minimization on a known problem
        f = ExtendedQuadratic(np.array(
            [[1, 0], [0, -1]]), np.array([0, 0]), 0, np.array([[0, 1]]), np.array([0]))
        h, A, b = f.partial_minimization([0])
        assert h.convex
        assert h.P == -1
        assert h.q == 0
        assert h.r == 0
        assert h.F == 1 or h.F == -1
        assert h.g == 0

    def test_dp_vs_lqr(self):
        A = np.random.randn(1, 1, 2, 2)
        B = np.random.randn(1, 1, 2, 10)
        c = np.zeros((1, 1, 2))
        g = [[ExtendedQuadratic(np.eye(12), np.zeros(12), 0)]]
        Pi = np.ones((1, 1))

        def sample(t):
            return A, B, c, g, Pi
        V, Q, policy = dp_infinite(sample, 100, 1)

        from scipy.linalg import solve_discrete_are

        Q = np.eye(2)
        R = np.eye(10)
        P = solve_discrete_are(A[0][0], B[0][0], Q, R)

        # check the value function
        assert np.allclose(V[0].P, P)

        # check the policy
        K = -np.linalg.solve(R + B[0][0].T@P@B[0][0], B[0][0].T@P@A[0][0])
        assert np.allclose(policy[0][0], K)


if __name__ == '__main__':
    unittest.main()
