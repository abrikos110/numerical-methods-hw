def second_order_Runge_Kutta(f, x0, y0, h, n):
    """y can also be a vector
    y' = f(x, y); y(x0) = y0"""
    x = x0
    y = y0
    if hasattr(x, 'copy'):
        x = x.copy()
    if hasattr(y, 'copy'):
        y = y.copy()
    for i in range(n):
        k1 = h * f(x, y)
        k2 = h * f(x + h, y + k1)
        y += (k1 + k2) / 2
        x += h
    return x, y


def fourth_order_Runge_Kutta(f, x0, y0, h, n):
    """y can also be a vector
    y' = f(x, y); y(x0) = y0"""
    x = x0
    y = y0
    if hasattr(x, 'copy'):
        x = x.copy()
    if hasattr(y, 'copy'):
        y = y.copy()
    for i in range(n):
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h
    return x, y


if __name__ == '__main__':
    from numpy import exp, sin, cos
    import numpy
    import matplotlib.pyplot as plt

    # таблица 1 - функция 6
    for f1, F1, xy1 in [(lambda x, y: (y - y**2) * x, lambda x: 1 / (1 - 2/3*exp(-x**2/2)), (0, 3)),
            (lambda x, y: 3-y-x, lambda x: 4 - x - 4*exp(-x), (0, 0)),
            (lambda x, y: (y - y*x), lambda x: 5 * exp(-0.5*x*(-2+x)), (0, 5)),
            (lambda x, y: sin(x) - y, lambda x: -0.5*cos(x) + 0.5*sin(x) + 21/2*exp(-x), (0, 10)),
            (lambda x, y: (-y - x**2), lambda x: -x**2 + 2*x - 2 + 12*exp(-x), (0, 10)),
            (lambda x, y: (x - x**2) * y, lambda x: exp(-1/6 * x**2 * (-3+2*x)), (0, 1))]:
        xs, ys = second_order_Runge_Kutta(f1, *xy1, 0.1, 10)
        print('error for second order method:', F1(1) - ys)
        xf, yf = fourth_order_Runge_Kutta(f1, *xy1, 0.1, 10)
        print('error for fourth order method:', F1(1) - yf)
        print()

    # таблица 2 - функция 21
    f2 = lambda x, y: numpy.array([2.4*y[1] - y[0], exp(-y[0]) - x + 2.2*y[1]])
    xy2 = (0, numpy.array([1, 0.25]))
    xs, ys = second_order_Runge_Kutta(f2, *xy2, 0.1, 10)
    xe, ye = fourth_order_Runge_Kutta(f2, *xy2, 0.0001, 10000)
    print('pseudo-error for second order method:', abs(ye - ys).max())
    xf, yf = fourth_order_Runge_Kutta(f2, *xy2, 0.1, 10)
    print('pseudo-error for fourth order method:', abs(ye - yf).max())


    xs, ys = xy1
    x, y = [xs], [ys]
    for i in range(10):
        xs, ys = second_order_Runge_Kutta(f1, xs, ys, 0.1, 1)
        x.append(xs)
        y.append(ys)
    plt.plot(x, y, label='метод второго порядка, $h=0.1$')

    xf, yf = xy1
    x, y = [xf], [yf]
    for i in range(10):
        xf, yf = fourth_order_Runge_Kutta(f1, xf, yf, 0.1, 1)
        x.append(xf)
        y.append(yf)
    plt.plot(x, y, label='метод четвертого порядка, $h=0.1$')
    plt.plot(numpy.linspace(0, 1, 100), F1(numpy.linspace(0, 1, 100)), label='точное решение')
    plt.legend()
    plt.show()


    xs, ys = [xy2[0]], [xy2[1]]
    for i in range(10):
        xy = second_order_Runge_Kutta(f2, xs[-1], ys[-1], 0.1, 1)
        xs.append(xy[0])
        ys.append(xy[1])
    plt.plot(xs, numpy.array(ys)[:, 0], label='$y_1$, второй порядок, $h=0.1$')
    plt.plot(xs, numpy.array(ys)[:, 1], label='$y_2$, второй порядок, $h=0.1$')

    xf, yf = [xy2[0]], [xy2[1]]
    for i in range(10):
        xy = fourth_order_Runge_Kutta(f2, xf[-1], yf[-1], 0.1, 1)
        xf.append(xy[0])
        yf.append(xy[1])
    plt.plot(xf, numpy.array(yf)[:, 0], label='$y_1$, четвертый порядок, $h=0.1$')
    plt.plot(xf, numpy.array(yf)[:, 1], label='$y_2$, четвертый порядок, $h=0.1$')

    xe, ye = [xy2[0]], [xy2[1]]
    for i in range(10**4):
        xy = fourth_order_Runge_Kutta(f2, xe[-1], ye[-1], 10**-4, 1)
        xe.append(xy[0])
        ye.append(xy[1])
    plt.plot(xe, numpy.array(ye)[:, 0], label='$y_1$, четвертый порядок, $h=10^{-4}$')
    plt.plot(xe, numpy.array(ye)[:, 1], label='$y_2$, четвертый порядок, $h=10^{-4}$')
    plt.legend()
    plt.show()
