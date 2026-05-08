def secant_method(f_prime, a, b, iterations=3):
    k = 1
    points = []
    f1 = f_prime(a)
    f2 = f_prime(b)

    if f1 * f2 > 0:
        raise ValueError(
            "secant_method потребує інтервал, де f'(a) та f'(b) мають різні знаки."
        )

    while k <= iterations:
        f1_curr = f1
        f2_curr = f2
        denom = f2_curr - f1_curr
        if denom == 0:
            print(
                f"[{k}] x1={a:.3f} x2={b:.3f} | f'(x1)={f1_curr:.3f} "
                f"f'(x2)={f2_curr:.3f} | стоп"
            )
            break

        x_star = b - f2_curr * (b - a) / denom
        f_star = f_prime(x_star)
        points.append((k, a, b, x_star, f1_curr, f2_curr, f_star))

        if f_star == 0:
            print(
                f"[{k}] x1={points[-1][1]:.3f} x2={points[-1][2]:.3f} "
                f"x*={points[-1][3]:.3f} | f'(x1)={f1_curr:.3f} f'(x2)={f2_curr:.3f} "
                f"f'(x*)={f_star:.3f} | викл.=нема (точний корінь)"
            )
            a = x_star
            b = x_star
            break

        if f1_curr * f_star <= 0:
            excluded = (x_star, b)
            b = x_star
            f2 = f_star
        else:
            excluded = (a, x_star)
            a = x_star
            f1 = f_star

        print(
            f"[{k}] x1={points[-1][1]:.3f} x2={points[-1][2]:.3f} "
            f"x*={points[-1][3]:.3f} | f'(x1)={f1_curr:.3f} f'(x2)={f2_curr:.3f} "
            f"f'(x*)={f_star:.3f} | викл.=[{excluded[0]:.3f}; {excluded[1]:.3f}]"
        )
        k += 1

    return (a, b), points
