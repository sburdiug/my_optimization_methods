import numpy as np


def rk4(f, x0, y0, xn, h):
    steps = int((xn - x0) / h)
    xs = x0 + h * np.arange(steps + 1, dtype=float)

    y = np.asarray(y0, dtype=float)
    ys = np.empty((steps + 1,) + y.shape, dtype=float)
    ys[0] = y

    for i in range(steps):
        x = xs[i]
        k1 = h * np.asarray(f(x, y), dtype=float)
        k2 = h * np.asarray(f(x + h / 2.0, y + k1 / 2.0), dtype=float)
        k3 = h * np.asarray(f(x + h / 2.0, y + k2 / 2.0), dtype=float)
        k4 = h * np.asarray(f(x + h, y + k3), dtype=float)
        y = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        ys[i + 1] = y

    return xs, ys
