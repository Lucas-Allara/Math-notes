"""Gradient descent with fixed step size and averaging of the iterates.

Section 11.2 of the Linear Algebra Notes (convergence theorem for
convex functions). It is the core of the optimizers in machine learning
frameworks, which use variants: minibatch stochastic gradient,
momentum, adaptive steps (Adam); here the step is fixed and
the average x̄ is returned, which is the variant with the guarantee of the theorem.
"""


def gradient_descent(gradient, x0, eta, T):
    """Iterates x <- x - eta * gradient(x) for T steps.

    Returns (average, iterates): the average of x^(0), ..., x^(T-1) (the x̄
    of the theorem, with f(x̄) - f(x*) <= RL/sqrt(T) for the optimal eta) and the
    full list of iterates x^(0), ..., x^(T)."""
    iterates = [list(x0)]
    x = list(x0)
    for _ in range(T):
        g = gradient(x)
        x = [xi - eta * gi for xi, gi in zip(x, g)]
        iterates.append(x)
    n = len(x0)
    average = [sum(it[i] for it in iterates[:T]) / T for i in range(n)]
    return average, iterates


def _tests():
    from fractions import Fraction as F
    # the example of section 11.2: f(x) = (1/2)||Ax - b||^2 with the A and b of
    # least squares; gradient A^T A x - A^T b, x0 = 0, eta = 1/4.
    # Exact iterates: (1/2, 1/2), (5/8, 5/8), (21/32, 21/32) -> (2/3, 2/3)
    def gradient(x):
        return [2 * x[0] + x[1] - 2, x[0] + 2 * x[1] - 2]

    _, its = gradient_descent(gradient, [F(0), F(0)], F(1, 4), 3)
    assert its[1] == [F(1, 2), F(1, 2)]
    assert its[2] == [F(5, 8), F(5, 8)]
    assert its[3] == [F(21, 32), F(21, 32)]
    # closed form of the example: x^(t) = 2/3 - (2/3) 4^(-t) in each coordinate
    for t, it in enumerate(its):
        assert it[0] == F(2, 3) - F(2, 3) / 4 ** t
    # with many iterations, f decreases toward f(x̂) = 1/6
    def f(x):
        r = [x[0] + x[1] - 1, x[0] - 1, x[1] - 1]
        return sum(ri ** 2 for ri in r) / 2

    _, its = gradient_descent(gradient, [0.0, 0.0], 0.25, 60)
    values = [f(x) for x in its]
    assert all(a >= b - 1e-15 for a, b in zip(values, values[1:]))
    assert abs(values[-1] - 1 / 6) < 1e-12
    print("gradient_descent: ok")


if __name__ == "__main__":
    _tests()
