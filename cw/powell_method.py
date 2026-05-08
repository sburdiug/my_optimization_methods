def powell_method(f, a, b, eps=0.01, max_iter=100, verbose=False):
    if eps <= 0:
        raise ValueError("eps має бути додатним.")
    if max_iter <= 0:
        raise ValueError("max_iter має бути додатним.")

    x1 = float(a)
    x3 = float(b)
    if x1 > x3:
        x1, x3 = x3, x1

    x2 = 0.5 * (x1 + x3)
    f1 = float(f(x1))
    f2 = float(f(x2))
    f3 = float(f(x3))
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
            x_star = 0.5 * (x1 + x3)

        f_star = float(f(x_star))
        x_min, f_min = min(((x1, f1), (x2, f2), (x3, f3), (x_star, f_star)), key=lambda item: item[1])
        dx = abs(x_star - x2)
        df = abs(f_min - f_star)
        criterion_met = dx <= eps or abs(x3 - x1) <= eps

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
                "criterion_met": criterion_met,
            }
        )

        if verbose:
            print(
                f"[{k}] x1={x1:.6g} x2={x2:.6g} x3={x3:.6g} x*={x_star:.6g} | "
                f"xmin={x_min:.6g} | |x*-xmin|={dx:.3e} | |f*-fmin|={df:.3e}"
            )

        if criterion_met:
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

        points = sorted(((x1, f1), (x2, f2), (x3, f3), (x_star, f_star)), key=lambda item: item[0])
        min_index = min(range(len(points)), key=lambda i: points[i][1])

        if min_index == 0:
            selected = points[:3]
        elif min_index == len(points) - 1:
            selected = points[-3:]
        else:
            selected = points[min_index - 1 : min_index + 2]

        (x1, f1), (x2, f2), (x3, f3) = selected

        if abs(x3 - x1) <= eps:
            x_min, f_min = min(((x1, f1), (x2, f2), (x3, f3)), key=lambda item: item[1])
            return x_min, {
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "x_star": x_min,
                "x_min": x_min,
                "f_min": f_min,
                "dx": abs(x3 - x1),
                "df": 0.0,
                "criterion_met": True,
                "iterations": k + 1,
                "history": history,
            }

    x_min, f_min = min(((x1, f1), (x2, f2), (x3, f3)), key=lambda item: item[1])
    return x_min, {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x_star": x_min,
        "x_min": x_min,
        "f_min": f_min,
        "dx": abs(x3 - x1),
        "df": 0.0,
        "criterion_met": False,
        "iterations": max_iter,
        "history": history,
    }
