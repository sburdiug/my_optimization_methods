def powell_method(f, a, b, eps=0.01):
    x1 = a
    x2 = (a + b) / 2
    x3 = b
    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)
    k = 0
    tiny = 1e-12

    while (x3 - x1) > eps:
        denom = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
        if abs(denom) <= tiny:
            x_new = x2
        else:
            num = (x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)
            x_new = x2 - 0.5 * num / denom

        if x_new <= x1 or x_new >= x3:
            x_new = x2

        print(f"[{k}] x1={x1:.3f} x2={x2:.3f} x3={x3:.3f} | x_new={x_new:.3f} | L={x3 - x1:.3f}")

        if abs(x_new - x2) <= tiny:
            return (x2, x2)

        f_new = f(x_new)

        if x_new > x2:
            if f_new >= f2:
                x3 = x_new
                f3 = f_new
            else:
                x1 = x2
                f1 = f2
                x2 = x_new
                f2 = f_new
        else:
            if f_new >= f2:
                x1 = x_new
                f1 = f_new
            else:
                x3 = x2
                f3 = f2
                x2 = x_new
                f2 = f_new

        if not (x1 < x2 < x3):
            xs = [(x1, f1), (x2, f2), (x3, f3)]
            xs.sort(key=lambda item: item[0])
            (x1, f1), (x2, f2), (x3, f3) = xs

        k += 1

    return (x1, x3)
