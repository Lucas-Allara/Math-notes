"""Least squares: by the normal equations and by the QR route.

Section 10.4 of the Linear Algebra Notes (theorem of the normal
equations and the remark on QR).
Equivalent: ``np.linalg.lstsq``. Differences: numpy solves with an SVD and
handles rank deficiency; here linearly independent columns are required.
The QR route avoids forming A^T A (better conditioned); the normal-equations
one is included because it is the one in the theorem.
"""
from lup import solve
from qr_gram_schmidt import qr_factorization, _col, _dot


def normal_equations(A, b):
    """Solves A^T A x = A^T b (Theorem «Normal Equations»)."""
    n = len(A[0])
    AtA = [[_dot(_col(A, i), _col(A, j)) for j in range(n)] for i in range(n)]
    Atb = [_dot(_col(A, i), b) for i in range(n)]
    return solve(AtA, Atb)


def least_squares(A, b):
    """QR route: R x = Q^T b, solved by back substitution."""
    Q, R = qr_factorization(A)
    n = len(R)
    qtb = [_dot(_col(Q, i), b) for i in range(n)]
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (qtb[i] - sum(R[i][j] * x[j] for j in range(i + 1, n))) / R[i][i]
    return x


def _tests():
    import random
    # the example of section 10.4: x̂ = (2/3, 2/3), residual (−1/3, 1/3, 1/3)
    A = [[1, 1], [1, 0], [0, 1]]
    b = [1, 1, 1]
    for method in (normal_equations, least_squares):
        x = method(A, b)
        assert abs(x[0] - 2 / 3) < 1e-12 and abs(x[1] - 2 / 3) < 1e-12
    # the residual is orthogonal to the columns of A (characterization of the projection)
    x = least_squares(A, b)
    residual = [b[i] - sum(A[i][j] * x[j] for j in range(2)) for i in range(3)]
    for j in range(2):
        assert abs(_dot(_col(A, j), residual)) < 1e-12
    # random: both routes agree
    random.seed(3)
    for _ in range(20):
        m = random.randint(3, 7)
        n = random.randint(1, m - 1)
        A = [[random.uniform(-5, 5) for _ in range(n)] for _ in range(m)]
        b = [random.uniform(-5, 5) for _ in range(m)]
        x1, x2 = normal_equations(A, b), least_squares(A, b)
        assert all(abs(a - c) < 1e-8 for a, c in zip(x1, x2))
    print("least_squares: ok")


if __name__ == "__main__":
    _tests()
