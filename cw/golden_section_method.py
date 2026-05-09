import numpy as np


def golden_section_method(f, a, b, eps=1e-3, max_iter=300, verbose=False):
    if eps <= 0:
        raise ValueError("eps має бути додатним.")
    if max_iter <= 0:
        raise ValueError("max_iter має бути додатним.")

    a = float(a)
    b = float(b)
    if a > b:
        a, b = b, a

    tau = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = b - tau * (b - a)
    x2 = a + tau * (b - a)
    f1 = float(f(x1))
    f2 = float(f(x2))
    history = []

    for k in range(max_iter):
        length = abs(b - a)
        history.append(
            {
                "k": k,
                "a": a,
                "b": b,
                "length": length,
                "x1": x1,
                "x2": x2,
                "f1": f1,
                "f2": f2,
                "criterion_met": length <= eps,
            }
        )

        if verbose:
            print(
                f"[{k}] a={a:.6g} b={b:.6g} L={length:.6g} | "
                f"x1={x1:.6g} x2={x2:.6g} | f1={f1:.6g} f2={f2:.6g}"
            )

        if length <= eps:
            break

        if f1 <= f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - tau * (b - a)
            f1 = float(f(x1))
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + tau * (b - a)
            f2 = float(f(x2))

    return (a, b), history
