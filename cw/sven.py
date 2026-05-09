import matplotlib.pyplot as plt
import numpy as np


def sven_method(f, x0, delta, max_iter=60, verbose=False):
    if delta == 0:
        raise ValueError("delta має бути ненульовим.")
    if max_iter <= 0:
        raise ValueError("max_iter має бути додатним.")

    x0 = float(x0)
    delta = abs(float(delta))
    history = []

    f0 = float(f(x0))
    x_minus = x0 - delta
    x_plus = x0 + delta
    f_minus = float(f(x_minus))
    f_plus = float(f(x_plus))

    history.append(
        {
            "k": 0,
            "x": x0,
            "f_x": f0,
            "x_minus": x_minus,
            "f_minus": f_minus,
            "x_plus": x_plus,
            "f_plus": f_plus,
            "direction": "start",
        }
    )

    if verbose:
        print(f"[k=0] x={x0:.6g} f={f0:.6g}")
        print(f"f(x0-delta)={f_minus:.6g}  f(x0+delta)={f_plus:.6g}")

    if f_minus >= f0 and f_plus >= f0:
        interval = (min(x_minus, x_plus), max(x_minus, x_plus))
        if verbose:
            print("Мінімум міститься між x0-delta та x0+delta.")
        return interval, history

    if f_minus < f0 < f_plus:
        direction = "left"
        step = -delta
        if verbose:
            print("Напрямок пошуку: ліворуч.")
    elif f_minus > f0 > f_plus:
        direction = "right"
        step = delta
        if verbose:
            print("Напрямок пошуку: праворуч.")
    else:
        interval = (min(x_minus, x_plus), max(x_minus, x_plus))
        if verbose:
            print("Напрямок не визначено, взято початковий інтервал.")
        return interval, history

    x_prev = x0
    x_curr = x0
    f_curr = f0

    for k in range(1, max_iter + 1):
        x_next = x_curr + (2 ** (k - 1)) * step
        f_next = float(f(x_next))
        history.append(
            {
                "k": k,
                "x_prev": x_prev,
                "x_curr": x_curr,
                "x_next": x_next,
                "f_curr": f_curr,
                "f_next": f_next,
                "direction": direction,
            }
        )

        if verbose:
            print(f"[k={k}] x={x_next:.6g} f={f_next:.6g}")

        if f_next > f_curr:
            interval = (min(x_prev, x_next), max(x_prev, x_next))
            if verbose:
                print(f"Функція зросла. Інтервал: [{interval[0]:.6g}, {interval[1]:.6g}]")
            return interval, history

        x_prev = x_curr
        x_curr = x_next
        f_curr = f_next

    interval = (min(x_prev, x_curr), max(x_prev, x_curr))
    if verbose:
        print(f"Досягнуто max_iter. Поточний інтервал: [{interval[0]:.6g}, {interval[1]:.6g}]")
    return interval, history


def visualize_sven(f, interval, history, x0, delta):
    """Візуалізація методу Свена."""
    a, b = interval
    xs = []
    fs = []
    for item in history:
        if "x" in item:
            xs.append(item["x"])
            fs.append(item["f_x"])
        if "x_next" in item:
            xs.append(item["x_next"])
            fs.append(item["f_next"])

    x_values = xs + [a, b]
    x_min = min(x_values) - 2
    x_max = max(x_values) + 2
    grid = np.linspace(x_min, x_max, 1000)
    values = [f(x) for x in grid]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(grid, values, "b-", linewidth=2, label="f(x)")
    ax.axvline(a, color="r", linestyle="--", alpha=0.7, label=f"a = {a:.3f}")
    ax.axvline(b, color="r", linestyle="--", alpha=0.7, label=f"b = {b:.3f}")

    for i, (x, fx) in enumerate(zip(xs, fs)):
        ax.plot(x, fx, "ro", markersize=7)
        ax.annotate(f"x_{i}", (x, fx), xytext=(5, 5), textcoords="offset points", fontsize=9)

    ax.plot(x0, f(x0), "go", markersize=10, label=f"x0 = {x0}")
    ax.fill_between([a, b], ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.2, color="yellow", label="Інтервал невизначеності")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Метод Свена для визначення інтервалу невизначеності")
    ax.legend()
    ax.set_xlim(x_min, x_max)
    plt.tight_layout()
    plt.show()
    return fig
