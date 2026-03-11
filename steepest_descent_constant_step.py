from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def steepest_descent_constant_step(
    fx: sp.Expr,
    x: Sequence[float],
    step: float,
    max_iter: int,
):
    """
    Метод найшвидшого спуску з постійним кроком:
    x^(k+1) = x^(k) - step * grad(f(x^(k))) / ||grad(f(x^(k)))||.
    Вхід: fx, x, step, max_iter
    """
    if step <= 0:
        raise ValueError("step має бути додатним.")
    if max_iter <= 0:
        raise ValueError("max_iter має бути >= 1.")

    variables = tuple(sorted(fx.free_symbols, key=lambda s: s.name))
    xk = np.asarray(x, dtype=float).reshape(-1)

    if len(variables) != xk.size:
        raise ValueError(
            "Кількість змінних у fx має збігатися з розмірністю x."
        )

    gradient_expr = [sp.diff(fx, var) for var in variables]
    f_func = sp.lambdify(variables, fx, modules="numpy")
    grad_func = sp.lambdify(variables, gradient_expr, modules="numpy")

    history = []

    for k in range(max_iter):
        grad_k = np.asarray(grad_func(*xk), dtype=float).reshape(-1)
        grad_norm_k = float(np.linalg.norm(grad_k))
        f_k = float(f_func(*xk))

        if grad_norm_k == 0:
            s_k = np.zeros_like(xk)
            x_next = xk.copy()
        else:
            s_k = -grad_k / grad_norm_k
            x_next = xk + float(step) * s_k

        grad_next = np.asarray(grad_func(*x_next), dtype=float).reshape(-1)
        grad_norm_next = float(np.linalg.norm(grad_next))
        f_next = float(f_func(*x_next))

        history.append(
            {
                "k": k,
                "x": xk.copy(),
                "f_x": f_k,
                "grad_x": grad_k.copy(),
                "grad_norm_x": grad_norm_k,
                "s": s_k.copy(),
                "x_next": x_next.copy(),
                "f_x_next": f_next,
                "grad_x_next": grad_next.copy(),
                "grad_norm_x_next": grad_norm_next,
            }
        )

        xk = x_next

        if grad_norm_k == 0:
            break

    grad_final = np.asarray(grad_func(*xk), dtype=float).reshape(-1)
    grad_norm_final = float(np.linalg.norm(grad_final))
    f_final = float(f_func(*xk))

    return {
        "fx": fx,
        "variables": variables,
        "gradient_expr": gradient_expr,
        "step": float(step),
        "max_iter": int(max_iter),
        "history": history,
        "x_final": xk,
        "f_final": f_final,
        "grad_final": grad_final,
        "grad_norm_final": grad_norm_final,
    }


def plot_steepest_descent_constant_step(result: dict):
    variables = result["variables"]
    if len(variables) != 2:
        raise ValueError("Побудова графіка підтримується лише для 2D функцій.")

    history = result["history"]
    if not history:
        return

    points = [np.asarray(history[0]["x"], dtype=float)]
    points.extend(np.asarray(item["x_next"], dtype=float) for item in history)
    points = np.vstack(points)

    x_vals = points[:, 0]
    y_vals = points[:, 1]
    x_center = 0.5 * (x_vals.min() + x_vals.max())
    y_center = 0.5 * (y_vals.min() + y_vals.max())
    span = max(x_vals.max() - x_vals.min(), y_vals.max() - y_vals.min(), 1.0)
    half_range = 0.65 * span

    x_min, x_max = x_center - half_range, x_center + half_range
    y_min, y_max = y_center - half_range, y_center + half_range

    grid_x = np.linspace(x_min, x_max, 300)
    grid_y = np.linspace(y_min, y_max, 300)
    X, Y = np.meshgrid(grid_x, grid_y)

    f_grid = sp.lambdify(variables, result["fx"], modules="numpy")
    Z = f_grid(X, Y)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contour(X, Y, Z, levels=20, cmap="Blues", alpha=0.5)
    ax.plot(points[:, 0], points[:, 1], "o-", color="tab:red", linewidth=2.0, markersize=5)

    for i, point in enumerate(points):
        ax.annotate(f"x^{i}", (point[0], point[1]), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("МНС з постійним кроком: траєкторія")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


def run_steepest_descent_constant_step(
    fx: sp.Expr,
    x_start: Sequence[float],
    step: float,
    max_iter: int,
    show_plot: bool = True,
):
    result = steepest_descent_constant_step(
        fx=fx,
        x=x_start,
        step=step,
        max_iter=max_iter,
    )

    print("Метод найшвидшого спуску з постійним кроком")
    print(f"f(x) = {sp.expand(fx)}")
    print(f"df/dx1 = {result['gradient_expr'][0]}")
    print(f"df/dx2 = {result['gradient_expr'][1]}")

    for item in result["history"]:
        k = item["k"]
        print()
        print(f"Ітерація {k + 1}:")
        print(f"x^({k}) = ({item['x'][0]:.3f}, {item['x'][1]:.3f})")
        print(f"f(x^({k})) = {item['f_x']:.3f}")
        print(f"grad f(x^({k})) = ({item['grad_x'][0]:.3f}, {item['grad_x'][1]:.3f})")
        print(f"||grad f(x^({k}))|| = {item['grad_norm_x']:.3f}")
        print(f"s^({k}) = ({item['s'][0]:.3f}, {item['s'][1]:.3f})")
        print(f"x^({k + 1}) = ({item['x_next'][0]:.3f}, {item['x_next'][1]:.3f})")
        print(f"f(x^({k + 1})) = {item['f_x_next']:.3f}")

    if show_plot:
        plot_steepest_descent_constant_step(result)

    return result
