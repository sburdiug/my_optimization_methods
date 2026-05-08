def bisection(f, alpha, beta, eps, max_iter=1000):

    a, b = alpha, beta
    fa, fb = f(a), f(b)
    history = []

    if fa * fb > 0:
        raise ValueError("На [a,b] ")

    for k in range(max_iter):
        c = (a + b) / 2
        fc = f(c)

        history.append((k, a, b, abs(b - a) / 2,abs(b - a) / 2 < eps))

        if abs(b - a) / 2 < eps:
            return c, k + 1, history


        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return (a + b) / 2, max_iter, history
