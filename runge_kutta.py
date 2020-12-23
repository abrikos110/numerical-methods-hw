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
    from numpy import exp
    import numpy
    # таблица 1 - функция 6
    f1 = lambda x, y: (x - x**2) * y
    F1 = lambda x: exp(-1/6 * x**2 * (-3+2*x))
    xy1 = (0, 1)
    xs, ys = second_order_Runge_Kutta(f1, *xy1, 0.01, 100)
    print('error for second order method:', F1(1) - ys)
    xf, yf = fourth_order_Runge_Kutta(f1, *xy1, 0.01, 100)
    print('error for fourth order method:', F1(1) - yf)

    # таблица 2 - функция 21
    f2 = lambda x, y: numpy.array([2.4*y[1] - y[0], exp(-y[0]) - x + 2.2*y[1]])
    xy2 = (0, numpy.array([1, 0.25]))
    xs, ys = second_order_Runge_Kutta(f2, *xy2, 0.01, 100)
    xs2, ys2 = second_order_Runge_Kutta(f2, *xy2, 0.0001, 10000)
    print('pseudo-error for second order method:', abs(ys2 - ys).max())
    xf, yf = fourth_order_Runge_Kutta(f2, *xy2, 0.01, 100)
    xf2, yf2 = fourth_order_Runge_Kutta(f2, *xy2, 0.0001, 10000)
    print('pseudo-error for fourth order method:', abs(yf2 - yf).max())
