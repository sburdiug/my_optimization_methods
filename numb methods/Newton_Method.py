def newton_method(f, df, x0, eps=1e-5, max_iter=100):
    """
    f  — функція f(x)
    df — похідна f'(x)
    x0 — початкове наблтження
    eps — точність
    max_iter — максимальна кілкість ітерацій
    """
    x = x0
    k = 0
    diff = float("inf")
    history = []

    while diff >= eps and k < max_iter:
        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            raise ZeroDivisionError("Похідна = 0, метод Ньютона зупинено")

        x_new = x - fx / dfx
        diff = abs(x_new - x)

        history.append((k, x, fx, dfx, x_new, diff < eps))

        x = x_new
        k += 1

    return x, k, history
