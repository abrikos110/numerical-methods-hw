import numpy


def tridiagonal_matrix_algorithm(A, B, C, f):
    n = f.shape[0]
    x = numpy.zeros(n)
    alpha = numpy.zeros(n)
    beta = numpy.zeros(n)
    alpha[1] = -C[0] / B[0]
    beta[1] = f[0] / B[0]
    for i in range(1, n-1):
        alpha[i+1] = -C[i] / (A[i-1] * alpha[i] + B[i])
        beta[i+1] = (f[i] - A[i-1] * beta[i]) / (A[i-1] * alpha[i] + B[i])
    x[n-1] = (f[n-1] - A[n-2] * beta[n-1]) / (B[n-1] + A[n-2] * alpha[n-1])
    for i in range(n-2, -1, -1):
        x[i] = alpha[i+1] * x[i+1] + beta[i+1]
    return x


def solve_boundary(p, q, f, ome1, ome2, gam1, gam2, del1, del2):
    """ome1 y(0) + gam1 y'(0) = del1
    ome2 y(1) + gam1 y'(1) = del2
    y'' + p(x) y' + q(x) y + f(x) = 0"""
