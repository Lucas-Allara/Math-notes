"""LUP factorization and its by-products: systems, determinant and inverse.

Section 10.3 of the Linear Algebra Notes.
Equivalents: ``np.linalg.solve``, ``np.linalg.det``, ``np.linalg.inv``,
``scipy.linalg.lu``. Difference: LAPACK factorizes by blocks and with a fine
stability analysis; here there is only partial pivoting by magnitude, as
the notes recommend. It works over any Python field (float,
Fraction), which allows exact tests.
"""
from fractions import Fraction


def lup_factorization(A):
    """Returns (L, U, perm) with PA = LU: L unit lower triangular, U upper
    triangular and perm the row permutation (perm[k] = original row at
    position k). At each step the pivot is the entry of largest absolute value."""
    n = len(A)
    U = [row[:] for row in A]
    L = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    perm = list(range(n))
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(U[i][k]))
        if U[p][k] == 0:
            raise ValueError("singular matrix")
        if p != k:
            U[k], U[p] = U[p], U[k]
            perm[k], perm[p] = perm[p], perm[k]
            for j in range(k):  # the multipliers travel with their rows
                L[k][j], L[p][j] = L[p][j], L[k][j]
        for i in range(k + 1, n):
            m = U[i][k] / U[k][k]  # multiplier
            L[i][k] = m
            for j in range(k, n):  # Schur complement, row i
                U[i][j] -= m * U[k][j]
    return L, U, perm


def forward_substitution(L, b):
    """Solves Ly = b reading the equations in order (L unit triangular)."""
    y = []
    for i in range(len(L)):
        y.append(b[i] - sum(L[i][j] * y[j] for j in range(i)))
    return y


def back_substitution(U, y):
    """Solves Ux = y from the last equation back to the first."""
    n = len(U)
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


def solve(A, b):
    """Solves Ax = b via PA = LU: it is equivalent to L(Ux) = Pb, two substitutions."""
    L, U, perm = lup_factorization(A)
    pb = [b[i] for i in perm]
    return back_substitution(U, forward_substitution(L, pb))


def determinant(A):
    """det(A) = (-1)^s times the product of the pivots (remark in section 10.3)."""
    try:
        _, U, perm = lup_factorization(A)
    except ValueError:
        return 0
    # sign of the permutation from its cycle decomposition
    s, seen = 0, [False] * len(perm)
    for i in range(len(perm)):
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        s += max(length - 1, 0)
    prod = 1
    for i in range(len(U)):
        prod *= U[i][i]
    return -prod if s % 2 else prod


def inverse(A):
    """The columns of the inverse are the solutions of A x_j = e_j."""
    n = len(A)
    L, U, perm = lup_factorization(A)
    cols = []
    for j in range(n):
        e = [1 if i == j else 0 for i in range(n)]
        pb = [e[i] for i in perm]
        cols.append(back_substitution(U, forward_substitution(L, pb)))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def _tests():
    import random
    F = Fraction
    # the example of section 10.3, exact with fractions
    A = [[F(1), F(2), F(0)], [F(3), F(4), F(4)], [F(5), F(6), F(3)]]
    b = [F(3), F(7), F(8)]
    assert solve(A, b) == [F(-7, 5), F(11, 5), F(3, 5)]
    assert determinant(A) == 10
    Ai = inverse(A)
    for i in range(3):
        for j in range(3):
            assert sum(A[i][k] * Ai[k][j] for k in range(3)) == (1 if i == j else 0)
    # the example without LU factorization in the notes: [[0,1],[1,1]] requires a permutation
    assert solve([[F(0), F(1)], [F(1), F(1)]], [F(2), F(3)]) == [F(1), F(2)]
    # PA = LU in the factorization with pivoting
    L, U, perm = lup_factorization(A)
    PA = [A[i] for i in perm]
    for i in range(3):
        for j in range(3):
            assert sum(L[i][k] * U[k][j] for k in range(3)) == PA[i][j]
    # random systems with floats: Ax ≈ b
    random.seed(1)
    for _ in range(30):
        n = random.randint(1, 6)
        M = [[random.uniform(-5, 5) for _ in range(n)] for _ in range(n)]
        c = [random.uniform(-5, 5) for _ in range(n)]
        x = solve(M, c)
        for i in range(n):
            assert abs(sum(M[i][k] * x[k] for k in range(n)) - c[i]) < 1e-9
    print("lup: ok")


if __name__ == "__main__":
    _tests()
