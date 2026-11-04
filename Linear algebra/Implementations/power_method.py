"""Power method with Rayleigh quotient, and deflation for symmetric matrices.

Section 10.5 of the Linear Algebra Notes.
Equivalent: ``np.linalg.eigh``. Differences: LAPACK uses the shifted QR
algorithm, which computes the whole spectrum at once and tolerates repeated
eigenvalues; the power method obtains the dominant one (and, with deflation, the first k
of a symmetric matrix), converges geometrically with ratio |λ2|/λ1 and requires
λ1 > |λ2|.
"""
import math
import random


def _matvec(A, x):
    return [sum(a * xi for a, xi in zip(row, x)) for row in A]


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def _normalize(v):
    norm = math.sqrt(_dot(v, v))
    return [vi / norm for vi in v]


def power_method(A, iterations=1000, x0=None, seed=0):
    """Returns (lambda, v): the dominant eigenvalue estimated by the Rayleigh
    quotient and its unit eigenvector. A random x0 has a nonzero component
    along v1 with probability 1 (remark in section 10.5)."""
    n = len(A)
    if x0 is None:
        rng = random.Random(seed)
        x0 = [rng.uniform(-1, 1) for _ in range(n)]
    x = _normalize(x0)
    for _ in range(iterations):
        x = _normalize(_matvec(A, x))
    return _dot(x, _matvec(A, x)), x


def symmetric_eigenpairs(A, k, iterations=1000, seed=0):
    """The k (eigenvalue, eigenvector) pairs with the largest eigenvalues of a
    symmetric matrix, by power method + deflation A' = A - lambda v v^T."""
    A = [row[:] for row in A]
    pairs = []
    for _ in range(k):
        lam, v = power_method(A, iterations, seed=seed)
        pairs.append((lam, v))
        for i in range(len(A)):
            for j in range(len(A)):
                A[i][j] -= lam * v[i] * v[j]
    return pairs


def _tests():
    from fractions import Fraction
    # the example of section 10.5: A = [[2,1],[1,2]], eigenvalues 3 and 1
    A = [[2, 1], [1, 2]]
    # unnormalized iterates from (1,0): (2,1), (5,4), (14,13), (41,40)
    x = [1, 0]
    for expected in ([2, 1], [5, 4], [14, 13], [41, 40]):
        x = _matvec(A, x)
        assert x == expected
    # exact Rayleigh quotients of the example: 2, 14/5, 122/41, 1094/365
    x = [1, 0]
    rayleighs = []
    for _ in range(4):
        rayleighs.append(Fraction(_dot(x, _matvec(A, x)), _dot(x, x)))
        x = _matvec(A, x)
    assert rayleighs == [Fraction(2), Fraction(14, 5), Fraction(122, 41), Fraction(1094, 365)]
    # convergence: lambda ≈ 3, v ≈ ±(1,1)/√2; deflation: the second pair ≈ (1, ±(1,-1)/√2)
    (l1, v1), (l2, v2) = symmetric_eigenpairs(A, 2, iterations=200)
    r2 = math.sqrt(2)
    assert abs(l1 - 3) < 1e-9 and abs(abs(v1[0]) - 1 / r2) < 1e-9 and abs(v1[0] - v1[1]) < 1e-9
    assert abs(l2 - 1) < 1e-9 and abs(abs(v2[0]) - 1 / r2) < 1e-9 and abs(v2[0] + v2[1]) < 1e-9
    # random symmetric 2x2 matrices against the quadratic formula
    rng = random.Random(5)
    for _ in range(20):
        a, c = rng.uniform(1, 5), rng.uniform(1, 5)
        b = rng.uniform(0.1, 0.9)
        M = [[a, b], [b, c]]
        expected = ((a + c) + math.sqrt((a - c) ** 2 + 4 * b * b)) / 2
        lam, v = power_method(M, iterations=2000)
        assert abs(lam - expected) < 1e-9
        # A v ≈ lambda v
        Av = _matvec(M, v)
        assert all(abs(avi - lam * vi) < 1e-6 for avi, vi in zip(Av, v))
    print("power_method: ok")


if __name__ == "__main__":
    _tests()
