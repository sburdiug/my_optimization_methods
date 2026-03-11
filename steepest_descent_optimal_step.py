from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def _compute_optimal_step(
    fx: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    xk: np.ndarray,
    s_k: np.ndarray,
) -> tuple[float, sp.Expr, sp.Expr]:
    lam = sp.symbols("lam", real=True)
    substitution = {
        var: float(xk[i]) + lam * float(s_k[i]) for i, var in enumerate(variables)
    }

    phi_expr = sp.expand(fx.subs(substitution))
    dphi_expr = sp.diff(phi_expr, lam)

    roots = sp.solve(sp.Eq(dphi_expr, 0), lam)
    candidates = []
    for root in roots:
        root_complex = complex(sp.N(root))
        if abs(root_complex.imag) < 1e-10:
            root_real = float(root_complex.real)
            if root_real >= 0:
                candidates.append(root_real)

    if not candidates:
        candidates = [0.0]

    phi_func = sp.lambdify(lam, phi_expr, modules="numpy")
    lambda_opt = min(candidates, key=lambda value: float(phi_func(value)))
    return lambda_opt, phi_expr, dphi_expr


def steepest_descent_optimal_step(
    fx: sp.Expr,
    x: Sequence[float],
    max_iter: int,
    eps: float = 1e-4,
):
    """
    Метод найшвидшого спуску з оптимальним кроком:
    s^(k) = -grad(f(x^(k)))
    lambda_k = argmin f(x^(k) + lambda*s^(k))
    x^(k+1) = x^(k) + lambda_k*s^(k)
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
    f_func = sp.lambdify(variables, fx, modules="numpy")
    grad_func = sp.lambdify(variables, gradient_expr, modules="numpy")

    history = []

    for k in range(max_iter):
        grad_k = np.asarray(grad_func(*xk), dtype=float).reshape(-1)
        grad_norm_k = float(np.linalg.norm(grad_k))
        f_k = float(f_func(*xk))
        s_k = -grad_k

        if grad_norm_k <= eps:
            history.append(
                {
                    "k": k,
                    "x": xk.copy(),
                    "f_x": f_k,
                    "grad_x": grad_k.copy(),
                    "grad_norm_x": grad_norm_k,
                    "s": s_k.copy(),
                    "lambda_opt": 0.0,
                    "x_next": xk.copy(),
                    "f_x_next": f_k,
                    "phi_expr": sp.Integer(0),
                    "dphi_expr": sp.Integer(0),
                }
            )
            break

        lambda_opt, phi_expr, dphi_expr = _compute_optimal_step(
            fx=fx, variables=variables, xk=xk, s_k=s_k
        )
        x_next = xk + lambda_opt * s_k
        f_next = float(f_func(*x_next))

        history.append(
            {
                "k": k,
                "x": xk.copy(),
                "f_x": f_k,
                "grad_x": grad_k.copy(),
                "grad_norm_x": grad_norm_k,
                "s": s_k.copy(),
                "lambda_opt": lambda_opt,
                "x_next": x_next.copy(),
                "f_x_next": f_next,
                "phi_expr": phi_expr,
                "dphi_expr": dphi_expr,
            }
        )

        xk = x_next

    grad_final = np.asarray(grad_func(*xk), dtype=float).reshape(-1)
    grad_norm_final = float(np.linalg.norm(grad_final))
    f_final = float(f_func(*xk))

    return {
        "fx": fx,
        "variables": variables,
        "gradient_expr": gradient_expr,
        "max_iter": int(max_iter),
        "eps": float(eps),
        "history": history,
        "x_final": xk,
        "f_final": f_final,
        "grad_final": grad_final,
        "grad_norm_final": grad_norm_final,
    }


def plot_steepest_descent_optimal_step(result: dict):
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
    ax.contour(X, Y, Z, levels=20, cmap="Greens", alpha=0.5)
    ax.plot(points[:, 0], points[:, 1], "o-", color="tab:purple", linewidth=2.0, markersize=5)

    for i, point in enumerate(points):
        ax.annotate(f"x^{i}", (point[0], point[1]), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("МНС з оптимальним кроком: траєкторія")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


def run_steepest_descent_optimal_step(
    fx: sp.Expr,
    x_start: Sequence[float],
    max_iter: int,
    eps: float = 1e-4,
    show_plot: bool = True,
):
    result = steepest_descent_optimal_step(
        fx=fx,
        x=x_start,
        max_iter=max_iter,
        eps=eps,
    )

    print("Метод найшвидшого спуску з оптимальним кроком")
    print(f"f(x) = {sp.expand(fx)}")
    for i, grad_i in enumerate(result["gradient_expr"], start=1):
        print(f"df/dx{i} = {grad_i}")

    for item in result["history"]:
        k = item["k"]
        print()
        print(f"Ітерація {k + 1}:")
        print(f"x^({k}) = ({item['x'][0]:.3f}, {item['x'][1]:.3f})")
        print(f"f(x^({k})) = {item['f_x']:.3f}")
        print(f"grad f(x^({k})) = ({item['grad_x'][0]:.3f}, {item['grad_x'][1]:.3f})")
        print(f"||grad f(x^({k}))|| = {item['grad_norm_x']:.3f}")
        print(f"s^({k}) = ({item['s'][0]:.3f}, {item['s'][1]:.3f})")
        print(f"lambda_{k} = {item['lambda_opt']:.6f}")
        print(f"x^({k + 1}) = ({item['x_next'][0]:.3f}, {item['x_next'][1]:.3f})")
        print(f"f(x^({k + 1})) = {item['f_x_next']:.3f}")

    if show_plot:
        plot_steepest_descent_optimal_step(result)

    return result


if __name__ == "__main__":
    x1, x2 = sp.symbols("x1 x2")
    fx = 4 * x1**2 + x1 * x2 + x2**2
    x_start = (10.0, 10.0)

    run_steepest_descent_optimal_step(
        fx=fx,
        x_start=x_start,
        max_iter=3,
    )
