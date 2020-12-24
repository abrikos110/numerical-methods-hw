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


def solve_boundary_0_1(p, q, f, ome1, ome2, gam1, gam2, del1, del2, n):
    """ome1 y(0) + gam1 y'(0) = del1
    ome2 y(1) + gam1 y'(1) = del2
    y'' + p(x) y' + q(x) y + f(x) = 0"""
    # перепутал местами при выводе формул
    ome1, gam1 = gam1, ome1
    ome2, gam2 = gam2, ome2
    A = 1e+100*numpy.ones(n-1)
    B = 1e+100*numpy.ones(n)
    C = 1e+100*numpy.ones(n-1)
    F = 1e+100*numpy.ones(n)
    h = 1/(n+1)
    for i in range(1, n-1):
        x = h*i
        A[i-1] = 1/h**2 - p(x)/2/h
        B[i] = -2/h**2 + q(x)
        C[i] = 1/h**2 + p(x)/2/h
        F[i] = -f(x)
    tau = (-1 + p(0) * h/2) ** -1
    F[0] = del1 + ome1 * tau * f(0) * h/2
    B[0] = gam1 + ome1 * tau * (1/h - q(0)*h/2)
    C[0] = -ome1 * tau / h

    tau = (-1 + p(1) * -h/2) ** -1
    F[n-1] = del2 + ome2 * tau * (f(1)+f(1-h)) * -h/4
    B[n-1] = gam2 + ome2 * tau * (1/-h - q(1)*-h/2)
    A[n-2] = -ome2 * tau / -h

    x = tridiagonal_matrix_algorithm(A, B, C, F)
    return x


def solve_boundary_a_b(p, q, f, ome1, ome2, gam1, gam2, del1, del2, n, a, b):
    # u(x) = y(x * (b-a) + a)
    # y(x) = u((x-a)/(b-a))
    # y'(x) = u'((x-a)/(b-a)) * 1/(b-a)
    # y''(x) = u''((x-a)/(b-a)) * 1/(b-a)**2

    # y'' + p y' + q y + f = 0 (x in [a:b])
    #   <=> 1/(b-a)**2 u'' + 1/(b-a) u' p + q u + f = 0 (x in [0 : 1])
    #   <=> u'' + (b-a) u' p + (b-a)**2 q u + (b-a)**2 f = 0 (x in [0 : 1])
    # y'(a) = u'(0) / (b-a)
    # y'(b) = u'(1) / (b-a)

    return solve_boundary_0_1(lambda x: (b-a) * p(x*(b-a) + a),
            lambda x: (b-a)**2 * q(x*(b-a) + a),
            lambda x: (b-a)**2 * f(x*(b-a) + a),
            ome1, ome2,
            gam1 / (b-a), gam2 / (b-a),
            del1, del2,
            n)


if __name__ == '__main__':
    import numpy
    from numpy import exp, sin, cos, sqrt
    import matplotlib.pyplot as plt

    # y'' - y'/2 + 3 y - 3x**2 = 0
    # y(1) - 2 y'(1) = 0.6
    # y(1.3) = 1

    F = lambda x: (2/3 * (x**2 + x/3 + 0.704467 * exp(x/4) * sin(sqrt(47) *x/4)
            + 0.927242 * exp(x/4) * cos(sqrt(47) *x/4) - 11/18))
    a, b = 1, 1.3
    ye = solve_boundary_a_b(lambda x: -1/2,
            lambda x: 3,
            lambda x: -2*x**2,
                1, 1,
                -2, 0,
                0.6, 1, 10**6, a, b)
    for N in [99, 9999]:
        x = numpy.linspace(a, b, N)
        y = solve_boundary_a_b(lambda x: -1/2,
                lambda x: 3,
                lambda x: -2*x**2,
                    1, 1,
                -2, 0,
                0.6, 1, N, a, b)
        # "exaxt" solution by wolframalpha
        print(abs(y-F(x)).max())
        plt.plot(x, ( y),
                label='численное решение при $h=\\frac{{{0}}}{{{1}}}$'.format(round(b-a, 10), N+1))
    plt.legend()
    plt.show()
