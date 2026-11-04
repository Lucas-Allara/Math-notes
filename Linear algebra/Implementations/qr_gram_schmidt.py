"""QR factorization by the classical Gram-Schmidt process.

Section 10.4 of the Linear Algebra Notes (existence and uniqueness theorem).
Equivalent: ``np.linalg.qr``. Important difference: numpy uses Householder
reflections, which are numerically stable; classical Gram-Schmidt loses
orthogonality when the columns are nearly dependent (and numpy does not require
a positive diagonal in R, so its signs may differ from these).
"""
import math


def _col(A, j):
    return [row[j] for row in A]


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def qr_factorization(A):
    """A (m×n, linearly independent columns) = QR, with Q having orthonormal
    columns and R upper triangular with positive diagonal.

    The formulas are those of the theorem: r_jk = <a_k, q_j> for j < k and
    r_kk = ||v_k||, with v_k the Gram-Schmidt vector."""
    m, n = len(A), len(A[0])
    qs = []  # columns of Q
    R = [[0] * n for _ in range(n)]
    for k in range(n):
        a_k = _col(A, k)
        v = a_k[:]
        for j in range(k):
            R[j][k] = _dot(a_k, qs[j])
            v = [vi - R[j][k] * qj for vi, qj in zip(v, qs[j])]
        R[k][k] = math.sqrt(_dot(v, v))
        if R[k][k] < 1e-12:
            raise ValueError("linearly dependent columns")
        qs.append([vi / R[k][k] for vi in v])
    Q = [[qs[j][i] for j in range(n)] for i in range(m)]
    return Q, R


def _tests():
    import random
    # the example of section 10.4
    A = [[1, 1], [1, 0], [0, 1]]
    Q, R = qr_factorization(A)
    r2, r6 = math.sqrt(2), math.sqrt(6)
    expected_R = [[r2, 1 / r2], [0, r6 / 2]]
    expected_Q = [[1 / r2, 1 / r6], [1 / r2, -1 / r6], [0, 2 / r6]]
    for i in range(2):
        for j in range(2):
            assert abs(R[i][j] - expected_R[i][j]) < 1e-12
    for i in range(3):
        for j in range(2):
            assert abs(Q[i][j] - expected_Q[i][j]) < 1e-12
    # random: QR = A, Q^T Q = I, R triangular with positive diagonal
    random.seed(2)
    for _ in range(20):
        m = random.randint(2, 6)
        n = random.randint(1, m)
        A = [[random.uniform(-5, 5) for _ in range(n)] for _ in range(m)]
        Q, R = qr_factorization(A)
        for i in range(m):
            for j in range(n):
                qr = sum(Q[i][k] * R[k][j] for k in range(n))
                assert abs(qr - A[i][j]) < 1e-9
        for i in range(n):
            for j in range(n):
                qtq = _dot(_col(Q, i), _col(Q, j))
                assert abs(qtq - (1 if i == j else 0)) < 1e-9
                if i > j:
                    assert R[i][j] == 0
            assert R[i][i] > 0
    print("qr_gram_schmidt: ok")


if __name__ == "__main__":
    _tests()
