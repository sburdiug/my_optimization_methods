def powell_method(f, a, b, eps=0.01, max_iter=100):
    x1 = float(a)
    x2 = 0.5 * (a + b)
    x3 = float(b)
    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)
    tiny = 1e-12
    history = []

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
        x_min, f_min = min(((x1, f1), (x2, f2), (x3, f3)), key=lambda item: item[1])
        dx = abs(x_min - x_star)
        df = abs(f_min - f_star)
        history.append(
            {
                "k": k,
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "x_star": x_star,
                "x_min": x_min,
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "f_star": f_star,
                "f_min": f_min,
                "dx": dx,
                "df": df,
            }
        )
        print(
            f"[{k}] x1={x1:.3f} x2={x2:.3f} x3={x3:.3f} x*={x_star:.3f} | "
            f"xmin={x_min:.3f} | |x*-xmin|={dx:.3e} | |f*-fmin|={df:.3e}"
        )

        # Критерій завершення ДСК-Пауелла.
        if dx <= eps and df <= eps:
            return x_star, {
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "x_star": x_star,
                "x_min": x_min,
                "f_min": f_min,
                "dx": dx,
                "df": df,
                "criterion_met": True,
                "iterations": k + 1,
                "history": history,
            }

        if abs(x_star - x2) <= tiny:
            if abs(x_min - x1) <= tiny:
                x3 = x2
                f3 = f2
                x2 = 0.5 * (x1 + x3)
                f2 = f(x2)
            elif abs(x_min - x3) <= tiny:
                x1 = x2
                f1 = f2
                x2 = 0.5 * (x1 + x3)
                f2 = f(x2)
            else:
                return x2, {
                    "x1": x1,
                    "x2": x2,
                    "x3": x3,
                    "x_star": x2,
                    "x_min": x_min,
                    "f_min": f_min,
                    "dx": dx,
                    "df": df,
                    "criterion_met": False,
                    "iterations": k + 1,
                    "history": history,
                }
            continue

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

    x_min, f_min = min(((x1, f1), (x2, f2), (x3, f3)), key=lambda item: item[1])
    f_star = f(x2)
    return x2, {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x_star": x2,
        "x_min": x_min,
        "f_min": f_min,
        "dx": abs(x_min - x2),
        "df": abs(f_min - f_star),
        "criterion_met": False,
        "iterations": max_iter,
        "history": history,
    }
