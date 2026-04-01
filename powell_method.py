def powell_method(f, a, b, eps=0.01, max_iter=100):
    x1 = float(a)
    x2 = 0.5 * (a + b)
    x3 = float(b)
    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)
    tiny = 1e-12

    for k in range(max_iter):
        denom = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
        if abs(denom) <= tiny:
            x_star = x2
        else:
            num = (x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)
            x_star = x2 - 0.5 * num / denom

        if x_star <= x1 or x_star >= x3:
            x_star = x2

        f_star = f(x_star)
        dx = abs(x_star - x2)
        df = abs(f_star - f2)
        print(
            f"[{k}] x1={x1:.3f} x2={x2:.3f} x3={x3:.3f} x*={x_star:.3f} | "
            f"|x*-x2|={dx:.3e} | |f*-f2|={df:.3e}"
        )

        # Критерій завершення ДСК-Пауелла.
        if dx <= eps and df <= eps:
            return x_star, {
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "x_star": x_star,
                "dx": dx,
                "df": df,
                "criterion_met": True,
                "iterations": k + 1,
            }

        if x_star > x2:
            if f_star >= f2:
                x3 = x_star
                f3 = f_star
            else:
                x1 = x2
                f1 = f2
                x2 = x_star
                f2 = f_star
        else:
            if f_star >= f2:
                x1 = x_star
                f1 = f_star
            else:
                x3 = x2
                f3 = f2
                x2 = x_star
                f2 = f_star

        if not (x1 < x2 < x3):
            xs = [(x1, f1), (x2, f2), (x3, f3)]
            xs.sort(key=lambda item: item[0])
            (x1, f1), (x2, f2), (x3, f3) = xs

    return x2, {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x_star": x2,
        "dx": 0.0,
        "df": 0.0,
        "criterion_met": False,
        "iterations": max_iter,
    }
