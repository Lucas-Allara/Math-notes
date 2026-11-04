"""Principal Component Analysis by power method + deflation.

Sections 9 (PCA) and 10.5 (power method) of the Linear Algebra
Notes. Equivalent: ``sklearn.decomposition.PCA``. Differences: sklearn
computes an SVD of the centered data matrix, without forming the covariance
(numerically preferable, as the section on PCA via SVD points out); here
C = X^T X / (m-1) is formed and its dominant eigenvectors are extracted with the
power method, which is the construction in the notes.
"""
from power_method import symmetric_eigenpairs


def pca(data, k, iterations=2000):
    """Returns (components, variances, means): the first k principal
    components (as rows), the variance captured by each one (eigenvalue of
    the covariance matrix) and the column means used for centering."""
    m, n = len(data), len(data[0])
    means = [sum(row[j] for row in data) / m for j in range(n)]
    X = [[row[j] - means[j] for j in range(n)] for row in data]
    C = [[sum(X[t][i] * X[t][j] for t in range(m)) / (m - 1) for j in range(n)]
         for i in range(n)]
    pairs = symmetric_eigenpairs(C, k, iterations)
    components = [v for _, v in pairs]
    variances = [lam for lam, _ in pairs]
    return components, variances, means


def _tests():
    import math
    # centered data with covariance [[20/3, 16/3], [16/3, 20/3]]:
    # eigenvalues 12 and 4/3, components (1,1)/√2 and (1,-1)/√2
    data = [(3, 3), (-3, -3), (1, -1), (-1, 1)]
    components, variances, means = pca(data, 2)
    r2 = math.sqrt(2)
    assert all(abs(mu) < 1e-12 for mu in means)
    assert abs(variances[0] - 12) < 1e-9
    assert abs(variances[1] - 4 / 3) < 1e-9
    c1, c2 = components
    assert abs(abs(c1[0]) - 1 / r2) < 1e-9 and abs(c1[0] - c1[1]) < 1e-9
    assert abs(abs(c2[0]) - 1 / r2) < 1e-9 and abs(c2[0] + c2[1]) < 1e-9
    # the sum of the variances is the trace of the covariance (40/3)
    assert abs(sum(variances) - 40 / 3) < 1e-9
    print("pca: ok")


if __name__ == "__main__":
    _tests()
