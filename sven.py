import numpy as np
import matplotlib.pyplot as plt


def sven_method(f, x0, delta):
    k = 0
    points = [(k, x0, f(x0))]

    # Крок 1: Обчислюємо f(x₀)
    f0 = f(x0)

    # Крок 2: Обчислюємо f(x₀ - |Δ|) та f(x₀ + |Δ|)
    x_minus = x0 - abs(delta)
    x_plus = x0 + abs(delta)
    f_minus = f(x_minus)
    f_plus = f(x_plus)

    print(f"[k={k}] x={x0:.3f} f={f0:.3f}")
    print(f"f(x0-Δ)={f_minus:.3f}  f(x0+Δ)={f_plus:.3f}")

    # Визначаємо напрямок
    if f_minus > f0 > f_plus:
        # Рухаємось вправо
        print("Напрямок: вправо")
        delta = abs(delta)
        x_curr = x0
    elif f_minus < f0 < f_plus:
        # Рухаємось вліво
        print("Напрямок: вліво")
        delta = -abs(delta)
        x_curr = x0
    elif f_minus >= f0 and f_plus >= f0:
        # Мінімум між x_minus та x_plus
        print("Мінімум між x0-Δ та x0+Δ")
        return (min(x_minus, x_plus), max(x_minus, x_plus)), points
    else:
        # Некласичний випадок (плоскі ділянки або рівність)
        print("Невизначений напрямок, беремо інтервал навколо x0")
        return (min(x_minus, x_plus), max(x_minus, x_plus)), points

    # Крок 3: Ітеративне розширення інтервалу
    while True:
        x_next = x_curr + 2 ** k * delta
        f_next = f(x_next)
        points.append((k + 1, x_next, f_next))

        print(f"[k={k + 1}] x={x_next:.3f} f={f_next:.3f}")

        if f_next > f(x_curr):
            # Мінімум знайдено: додаємо проміжну точку
            print("f зросла, зупинка")
            x_mid = (x_curr + x_next) / 2
            f_mid = f(x_mid)
            points.append((k + 2, x_mid, f_mid))
            print(f"[k={k + 2}] x={x_mid:.3f} f={f_mid:.3f}")
            # Робимо x_mid центром інтервалу (а не граничною точкою).
            a = min(x_curr, x_next)
            b = max(x_curr, x_next)
            print(f"Інтервал: [{a:.3f}, {b:.3f}] (L={b - a:.3f})")
            return (a, b), points

        x_curr = x_next
        k += 1


def visualize_sven(f, interval, points, x0, delta):
    """Візуалізація методу Свена"""
    a, b = interval

    # Створюємо графік
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    # Графік 1: Функція та знайдені точки
    x_values = [p[1] for p in points] + [a, b]
    x_min = min(x_values) - 2
    x_max = max(x_values) + 2
    x = np.linspace(x_min, x_max, 1000)
    y = f(x)

    ax1.plot(x, y, 'b-', linewidth=2, label='f(x)')
    ax1.axvline(a, color='r', linestyle='--', alpha=0.7, label=f'a = {a:.3f}')
    ax1.axvline(b, color='r', linestyle='--', alpha=0.7, label=f'b = {b:.3f}')

    # Відмічаємо точки, знайдені методом Свена
    for i, (k, xk, fk) in enumerate(points):
        ax1.plot(xk, fk, 'ro', markersize=8)
        ax1.annotate(f'x_{k}', (xk, fk), xytext=(5, 5),
                     textcoords='offset points', fontsize=10)

    ax1.plot(x0, f(x0), 'go', markersize=12, label=f'x₀ = {x0}')
    ax1.fill_between([a, b], ax1.get_ylim()[0], ax1.get_ylim()[1],
                     alpha=0.2, color='yellow', label='Інтервал невизначеності')

    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Метод Свена для визначення інтервалу невизначеності', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(x_min, x_max)

    plt.tight_layout()
    plt.show()

    return fig
