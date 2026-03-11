def golden_section_method(f, a, b, eps=20):
    k = 0
    points = []

    while (b - a) > eps:
        x1 = a + 0.382 * (b - a)
        x2 = a + 0.618 * (b - a)
        f1 = f(x1)
        f2 = f(x2)
        points.append((k, a, b, x1, x2, f1, f2))

        print(
            f"[{k}] a={a:.3f} b={b:.3f} L={b - a:.3f} | "
            f"x1={x1:.3f} x2={x2:.3f} | f1={f1:.3f} f2={f2:.3f}"
        )

        if f1 < f2:
            b = x2
        else:
            a = x1

        k += 1

    return (a, b), points
