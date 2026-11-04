"""k-means with Lloyd's procedure.

Section 11.1 of the Linear Algebra Notes.
Equivalent: ``sklearn.cluster.KMeans``. Differences: sklearn initializes with
k-means++ and runs several restarts to avoid bad local minima; here
the initial centers are given explicitly (or sampled at random) and it is
run only once, which is the procedure in the notes.
"""
import random


def _delta(x, c):
    """Squared distance ||x - c||^2."""
    return sum((xi - ci) ** 2 for xi, ci in zip(x, c))


def _centroid(points):
    m = len(points)
    return tuple(sum(p[j] for p in points) / m for j in range(len(points[0])))


def kmeans(points, k, centers=None, seed=0):
    """Lloyd's procedure. Returns (clusters, centers, history):
    clusters as lists of point indices, and the cost f(S, C) after
    each recomputation of the centroids, which is nonincreasing and finite by Proposition
    «Monotonicity and Termination of Lloyd». Distance ties are broken
    by the smallest center index (the fixed rule in the notes)."""
    rng = random.Random(seed)
    centers = [tuple(c) for c in (centers if centers is not None else rng.sample(list(points), k))]
    history = []
    previous_assignment = None
    while True:
        # assign each point to a nearest center
        assignment = [min(range(k), key=lambda l: _delta(p, centers[l])) for p in points]
        if assignment == previous_assignment:
            break
        previous_assignment = assignment
        # recompute each center as the centroid of its cluster
        for l in range(k):
            group = [p for p, a in zip(points, assignment) if a == l]
            if group:
                centers[l] = _centroid(group)
        history.append(sum(_delta(p, centers[a]) for p, a in zip(points, assignment)))
    clusters = [[i for i, a in enumerate(assignment) if a == l] for l in range(k)]
    return clusters, centers, history


def _tests():
    # the example of section 11.1: S = {0, 2, 5, 6}, initial centers 0 and 2;
    # the point 2 changes cluster and the cost drops 26/3 -> 5/2 (optimal)
    points = [(0,), (2,), (5,), (6,)]
    clusters, centers, history = kmeans(points, 2, centers=[(0,), (2,)])
    assert clusters == [[0, 1], [2, 3]]
    assert centers == [(1.0,), (5.5,)]
    assert len(history) == 2
    assert abs(history[0] - 26 / 3) < 1e-12
    assert abs(history[1] - 5 / 2) < 1e-12
    # random: the cost never increases between recomputations
    rng = random.Random(6)
    for _ in range(10):
        pts = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(30)]
        _, _, hist = kmeans(pts, 3, seed=7)
        assert all(a >= b - 1e-9 for a, b in zip(hist, hist[1:]))
    print("kmeans_lloyd: ok")


if __name__ == "__main__":
    _tests()
