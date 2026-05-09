def euler(f, x0, y0, xn, h):
    steps = int((xn - x0) / h)
    x = x0
    y = y0
    xs = [x]
    ys = [y]

    for _ in range(steps):
        y = y + h * f(x, y)
        x = x + h
        xs.append(x)
        ys.append(y)

    return xs, ys

def euler_improved(f, x0, y0, xn, h):
    steps = int((xn - x0) / h)
    x = x0
    y = y0
    xs = [x]
    ys = [y]

    for _ in range(steps):
        y_pred = y + h * f(x, y)               # прогноз
        y = y + (h / 2) * (f(x, y) + f(x + h, y_pred))  # корекція
        x = x + h

        xs.append(x)
        ys.append(y)

    return xs, ys

# y' = x + y, y(0) = 1
def f(x, y):
    return 2*x - 3*y



