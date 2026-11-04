"""Matrix multiplication: direct and by blocks (divide and conquer).

Sections 10.1 and 10.2 of the Linear Algebra Notes.
numpy equivalent: ``A @ B``. Difference: numpy delegates to BLAS, which uses
cache-tuned blocks, vector instructions and parallelism; the
asymptotic cost is the same, the constant is not.
"""


def multiply(A, B):
    """Direct product by the definition, Theta(n·m·p) operations."""
    n, m, p = len(A), len(B), len(B[0])
    assert len(A[0]) == m, "incompatible dimensions"
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            s = 0
            for k in range(m):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def _add(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def _block(A, r0, r1, c0, c1):
    return [row[c0:c1] for row in A[r0:r1]]


def _assemble(C11, C12, C21, C22):
    top = [f1 + f2 for f1, f2 in zip(C11, C12)]
    bottom = [f1 + f2 for f1, f2 in zip(C21, C22)]
    return top + bottom


def multiply_blocks(A, B):
    """Recursive 2x2 block product (Proposition «Block Product»).

    Same number of operations as the direct one: T(n) = 8 T(n/2) + Theta(1)
    gives Theta(n^3); the point is that this partition is the basis of Strassen.
    """
    n = len(A)
    if n <= 2 or n % 2 == 1:
        return multiply(A, B)
    h = n // 2
    A11, A12 = _block(A, 0, h, 0, h), _block(A, 0, h, h, n)
    A21, A22 = _block(A, h, n, 0, h), _block(A, h, n, h, n)
    B11, B12 = _block(B, 0, h, 0, h), _block(B, 0, h, h, n)
    B21, B22 = _block(B, h, n, 0, h), _block(B, h, n, h, n)
    m = multiply_blocks
    C11 = _add(m(A11, B11), m(A12, B21))
    C12 = _add(m(A11, B12), m(A12, B22))
    C21 = _add(m(A21, B11), m(A22, B21))
    C22 = _add(m(A21, B12), m(A22, B22))
    return _assemble(C11, C12, C21, C22)


def _tests():
    import random
    random.seed(0)
    # blocks == direct on random integer matrices (exact comparison)
    for _ in range(20):
        n = random.choice([2, 4, 8])
        A = [[random.randint(-9, 9) for _ in range(n)] for _ in range(n)]
        B = [[random.randint(-9, 9) for _ in range(n)] for _ in range(n)]
        assert multiply(A, B) == multiply_blocks(A, B)
    # rectangular, against the hand computation
    A = [[1, 2], [3, 4], [5, 6]]
    B = [[1, 0, 2], [0, 1, 3]]
    assert multiply(A, B) == [[1, 2, 8], [3, 4, 18], [5, 6, 28]]
    print("block_product: ok")


if __name__ == "__main__":
    _tests()
