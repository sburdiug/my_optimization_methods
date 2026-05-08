def simple_iteration(phi, x0, eps=1e-5, max_iter=100):
    """
    phi       : функція φ(x), наприклад lambda x: ...
    x0        : початкове наближення
    eps       : точність
    max_iter  : максимум ітерацій
    return    : (корінь, кількість ітерацій, історія)
    """
    x = x0
    k = 0
    diff = float("inf")
    history = []

    while diff >= eps and k < max_iter:
        new = phi(x)
        diff = abs(new - x)
        history.append((k, x, new, diff, diff < eps))
        x = new
        k += 1

    return x, k, history

