def dichotomy_method(f, a, b, eps=10):
    tiny = 1e-12
    k = 0
    points = []

    while (b - a) - eps > tiny:
        xm = (a + b) / 2
        x1 = (a + xm) / 2
        x2 = (xm + b) / 2

        f_x1 = f(x1)
        f_xm = f(xm)
        f_x2 = f(x2)

        points.append((k, a, b, xm, x1, x2, f_x1, f_xm, f_x2))

        print(
            f"[k={k}] a={a:.3f} b={b:.3f} L={b-a:.3f} | "
            f"x1={x1:.3f} xm={xm:.3f} x2={x2:.3f} | "
            f"f1={f_x1:.3f} fm={f_xm:.3f} f2={f_x2:.3f}"
        )

        if f_x1 < f_xm:
            b = xm
        elif f_x2 < f_xm:
            a = xm
        else:
            a = x1
            b = x2

        k += 1

    return (a, b), points
