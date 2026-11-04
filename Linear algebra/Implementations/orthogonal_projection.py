"""Orthogonal projection onto the column space of a matrix.

Sections 2 (theory of the projection) and 10.4 (the identity P = QQ^T) of the
Linear Algebra Notes. It has no direct single-call equivalent in
numpy; it is the building block of least squares.
"""
from qr_gram_schmidt import qr_factorization, _col, _dot


def project(A, b):
    """P_{Col(A)}(b) = Q Q^T b, with A = QR the Gram-Schmidt factorization."""
    Q, _ = qr_factorization(A)
    n = len(Q[0])
    coefficients = [_dot(_col(Q, j), b) for j in range(n)]  # Q^T b
    return [sum(coefficients[j] * Q[i][j] for j in range(n)) for i in range(len(Q))]


def _tests():
    import random
    # the example of section 10.4: projection (4/3, 2/3, 2/3) and residual ⊥ columns
    A = [[1, 1], [1, 0], [0, 1]]
    b = [1, 1, 1]
    p = project(A, b)
    expected = [4 / 3, 2 / 3, 2 / 3]
    assert all(abs(pi - ei) < 1e-12 for pi, ei in zip(p, expected))
    residual = [bi - pi for bi, pi in zip(b, p)]
    for j in range(2):
        assert abs(_dot(_col(A, j), residual)) < 1e-12
    # idempotence: projecting twice is projecting once
    random.seed(4)
    for _ in range(20):
        m, n = 5, 2
        A = [[random.uniform(-5, 5) for _ in range(n)] for _ in range(m)]
        b = [random.uniform(-5, 5) for _ in range(m)]
        p1 = project(A, b)
        p2 = project(A, p1)
        assert all(abs(a - c) < 1e-9 for a, c in zip(p1, p2))
    print("orthogonal_projection: ok")


if __name__ == "__main__":
    _tests()
