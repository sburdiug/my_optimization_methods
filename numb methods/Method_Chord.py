def chord_method(f, alpha, beta, eps, max_iter=1000):
    """
    f         — функція f(x)
    a, β  — початкові межі відрізка [a, b]
    eps       — точність
    max_iter  — максимум ітерацій
    """
    a, b = alpha, beta
    fa, fb = f(a), f(b)
    x0 = a
    iteration = 0
    history = []

    while iteration < max_iter:
        iteration += 1

        # Формула хорд
        x = a - fa * (b - a) / (fb - fa)
        fx = f(x)

        # Вибір нового інтервалу
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx

        # Зберігаємо дані кроку
        history.append((iteration, a, b, abs(x - x0),abs(x - x0) < eps))

        # Перевірка умови зупинки
        if abs(x - x0) < eps:
            return x, iteration, history


        x0 = x

    return x, iteration, history