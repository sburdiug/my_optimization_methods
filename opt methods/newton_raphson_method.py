def newton_raphson_method(f, f_prime, f_double, x0, iterations=3):
    k = 0
    x = x0
    points = []

    while k < iterations:
        f1 = f_prime(x)
        f2 = f_double(x)
        if f2 == 0:
            print(f"[{k}] x={x:.3f} f'={f1:.3f} f''=0.000 | зупинка")
            break

        x_next = x - f1 / f2
        points.append((k, x, f(x), f1, f2))
        print(f"[{k}] x={x:.3f} f'={f1:.3f} f''={f2:.3f} | x_next={x_next:.3f}")

        x = x_next
        k += 1

    return x, points
