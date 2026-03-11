from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def _fmt_vec(values: np.ndarray, digits: int = 3) -> str:
    return "(" + ", ".join(f"{float(v):.{digits}f}" for v in values) + ")"


def newton_multivariate_method(
    fx: sp.Expr,
    x: Sequence[float],
    max_iter: int,
    eps: float = 1e-4,
):
    """
    Багатовимірний метод Ньютона:
    x^(k+1) = x^(k) - [H(x^(k))]^{-1} * grad(f(x^(k)))
    """
    if max_iter <= 0:
        raise ValueError("max_iter має бути >= 1.")
    if eps <= 0:
        raise ValueError("eps має бути додатним.")

    variables = tuple(sorted(fx.free_symbols, key=lambda s: s.name))
    xk = np.asarray(x, dtype=float).reshape(-1)

    if len(variables) != xk.size:
        raise ValueError("Кількість змінних у fx має збігатися з розмірністю x.")

    gradient_expr = [sp.diff(fx, var) for var in variables]
    hessian_expr = sp.hessian(fx, variables)

    f_func = sp.lambdify(variables, fx, modules="numpy")
    grad_func = sp.lambdify(variables, gradient_expr, modules="numpy")
    hess_func = sp.lambdify(variables, hessian_expr, modules="numpy")

    history = []
    points = [xk.copy()]

    for k in range(max_iter):
        grad_k = np.asarray(grad_func(*xk), dtype=float).reshape(-1)
        grad_norm_k = float(np.linalg.norm(grad_k))
        f_k = float(f_func(*xk))

        if grad_norm_k <= eps:
            history.append(
                {
                    "k": k,
                    "method": "stop",
                    "x": xk.copy(),
                    "f_x": f_k,
                    "grad_x": grad_k.copy(),
                    "grad_norm_x": grad_norm_k,
                    "delta": np.zeros_like(xk),
                    "delta_norm": 0.0,
                    "x_next": xk.copy(),
                    "f_x_next": f_k,
                }
            )
            break

        hessian_k = np.asarray(hess_func(*xk), dtype=float)

        try:
            delta = np.linalg.solve(hessian_k, grad_k)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(hessian_k) @ grad_k

        x_next = xk - delta
        delta_norm = float(np.linalg.norm(delta))
        f_next = float(f_func(*x_next))

        history.append(
            {
                "k": k,
                "method": "newton",
                "x": xk.copy(),
                "f_x": f_k,
                "grad_x": grad_k.copy(),
                "grad_norm_x": grad_norm_k,
                "hessian_x": hessian_k.copy(),
                "delta": delta.copy(),
                "delta_norm": delta_norm,
                "x_next": x_next.copy(),
                "f_x_next": f_next,
            }
        )

        xk = x_next
        points.append(xk.copy())

        if delta_norm <= eps:
            break

    grad_final = np.asarray(grad_func(*xk), dtype=float).reshape(-1)
    grad_norm_final = float(np.linalg.norm(grad_final))
    f_final = float(f_func(*xk))

    return {
        "fx": fx,
        "variables": variables,
        "gradient_expr": gradient_expr,
        "hessian_expr": hessian_expr,
        "max_iter": int(max_iter),
        "eps": float(eps),
        "history": history,
        "points": np.vstack(points),
        "x_final": xk,
        "f_final": f_final,
        "grad_final": grad_final,
        "grad_norm_final": grad_norm_final,
    }


def plot_newton_multivariate(result: dict):
    variables = result["variables"]
    if len(variables) != 2:
        raise ValueError("Побудова графіка підтримується лише для 2D функцій.")

    points = result["points"]
    if len(points) == 0:
        return

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
    ax.contour(X, Y, Z, levels=20, cmap="Purples", alpha=0.5)
    ax.plot(points[:, 0], points[:, 1], "o-", color="tab:green", linewidth=2.0, markersize=5)

    for i, point in enumerate(points):
        ax.annotate(f"x^{i}", (point[0], point[1]), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Метод Ньютона: траєкторія")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


def run_newton_multivariate_method(
    fx: sp.Expr,
    x_start: Sequence[float],
    max_iter: int,
    eps: float = 1e-4,
    show_plot: bool = True,
):
    result = newton_multivariate_method(
        fx=fx,
        x=x_start,
        max_iter=max_iter,
        eps=eps,
    )

    print("Метод Ньютона")
    print(f"f(x) = {sp.expand(fx)}")
    for i, grad_i in enumerate(result["gradient_expr"], start=1):
        print(f"df/dx{i} = {grad_i}")
    print(f"H(x) = {sp.Matrix(result['hessian_expr'])}")

    for item in result["history"]:
        k = item["k"]
        if item["method"] == "stop":
            print()
            print(f"Зупинка на ітерації {k + 1}:")
            print(f"x^({k}) = {_fmt_vec(item['x'])}")
            print(f"||grad f(x^({k}))|| = {item['grad_norm_x']:.6f} <= {result['eps']}")
            break

        print()
        print(f"Ітерація {k + 1} (Ньютон):")
        print(f"x^({k}) = {_fmt_vec(item['x'])}")
        print(f"f(x^({k})) = {item['f_x']:.3f}")
        print(f"grad f(x^({k})) = {_fmt_vec(item['grad_x'])}")
        print(f"||grad f(x^({k}))|| = {item['grad_norm_x']:.3f}")
        print(f"delta^({k}) = {_fmt_vec(item['delta'])}")
        print(f"x^({k + 1}) = {_fmt_vec(item['x_next'])}")
        print(f"f(x^({k + 1})) = {item['f_x_next']:.3f}")

    print()
    print(f"Фінальна точка: {_fmt_vec(result['x_final'], 6)}")
    print(f"f(x*) = {result['f_final']:.6f}")
    print(f"||grad f(x*)|| = {result['grad_norm_final']:.6f}")

    if show_plot:
        plot_newton_multivariate(result)

    return result


if __name__ == "__main__":
    x1, x2 = sp.symbols("x1 x2")
    fx = 4 * x1**2 + x1 * x2 + x2**2
    run_newton_multivariate_method(
        fx=fx,
        x_start=(10.0, 10.0),
        max_iter=3,
        show_plot=True,
    )
