from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def _compute_optimal_step_symbolic(
    fx: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    xk: np.ndarray,
    s_k: np.ndarray,
) -> float:
    """
    Обчислює оптимальний крок lambda_k вздовж напрямку s_k:
    phi(lambda) = f(xk + lambda * s_k) -> min, lambda >= 0.
    """
    lam = sp.symbols("lam", real=True)
    substitution = {
        var: float(xk[i]) + lam * float(s_k[i]) for i, var in enumerate(variables)
    }

    phi_expr = sp.expand(fx.subs(substitution))
    dphi_expr = sp.diff(phi_expr, lam)

    roots = sp.solve(dphi_expr, lam)
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
    return float(lambda_opt)


def _compute_optimal_step_quadratic(
    grad_k: np.ndarray,
    s_k: np.ndarray,
    hessian_const: np.ndarray,
) -> float:
    """
    Точний lambda_opt для квадратичної f:
    d/dlambda f(x + lambda*s) = grad^T s + lambda * s^T H s = 0
    => lambda = -(grad^T s) / (s^T H s)
    """
    denominator = float(s_k @ hessian_const @ s_k)
    if abs(denominator) <= 1e-14:
        return 0.0

    numerator = -float(grad_k @ s_k)
    lambda_opt = numerator / denominator
    return float(max(lambda_opt, 0.0))


def partan_mns(
    fx: sp.Expr,
    x: Sequence[float],
    max_iter: int,
    eps: float = 1e-4,
):
    """
    ПАРТАН-МНС на основі практики.

    Ідея:
    1) Перші дві ітерації - метод найшвидшого спуску з оптимальним кроком.
    2) Далі чергуються:
       - градієнтний крок (МНС): s^(k) = -grad f(x^(k))
       - партан-крок:            s^(k) = x^(k) - x^(k-2)
    3) Для кожного напрямку окремо шукається оптимальний крок lambda_k.
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

    hessian_expr = sp.hessian(fx, variables)
    has_variable_hessian = bool(hessian_expr.free_symbols)
    hessian_const = None
    if not has_variable_hessian:
        hessian_const = np.asarray(hessian_expr, dtype=float)

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
                    "s": np.zeros_like(xk),
                    "lambda_opt": 0.0,
                    "x_next": xk.copy(),
                    "f_x_next": f_k,
                }
            )
            break

        # Перші дві ітерації - МНС, далі чергування: ПАРТАН -> МНС -> ПАРТАН -> ...
        if k < 2:
            method_name = "mns"
            s_k = -grad_k
        else:
            if k % 2 == 0:
                method_name = "partan"
                s_k = xk - points[-3]
                if np.linalg.norm(s_k) <= 1e-12:
                    method_name = "mns"
                    s_k = -grad_k
            else:
                method_name = "mns"
                s_k = -grad_k

        if hessian_const is not None:
            lambda_opt = _compute_optimal_step_quadratic(
                grad_k=grad_k, s_k=s_k, hessian_const=hessian_const
            )
        else:
            lambda_opt = _compute_optimal_step_symbolic(
                fx=fx, variables=variables, xk=xk, s_k=s_k
            )

        if lambda_opt <= 1e-14 and method_name == "partan":
            method_name = "mns"
            s_k = -grad_k
            if hessian_const is not None:
                lambda_opt = _compute_optimal_step_quadratic(
                    grad_k=grad_k, s_k=s_k, hessian_const=hessian_const
                )
            else:
                lambda_opt = _compute_optimal_step_symbolic(
                    fx=fx, variables=variables, xk=xk, s_k=s_k
                )

        x_next = xk + lambda_opt * s_k
        f_next = float(f_func(*x_next))

        history.append(
            {
                "k": k,
                "method": method_name,
                "x": xk.copy(),
                "f_x": f_k,
                "grad_x": grad_k.copy(),
                "grad_norm_x": grad_norm_k,
                "s": s_k.copy(),
                "lambda_opt": lambda_opt,
                "x_next": x_next.copy(),
                "f_x_next": f_next,
            }
        )

        xk = x_next
        points.append(xk.copy())

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
        "points": np.vstack(points),
        "x_final": xk,
        "f_final": f_final,
        "grad_final": grad_final,
        "grad_norm_final": grad_norm_final,
    }


def plot_partan_mns(result: dict):
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
    ax.contour(X, Y, Z, levels=20, cmap="Oranges", alpha=0.5)
    ax.plot(points[:, 0], points[:, 1], "o-", color="tab:blue", linewidth=2.0, markersize=5)

    for i, point in enumerate(points):
        ax.annotate(f"x^{i}", (point[0], point[1]), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("ПАРТАН-МНС: траєкторія")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


def run_partan_mns(
    fx: sp.Expr,
    x_start: Sequence[float],
    max_iter: int,
    eps: float = 1e-4,
    show_plot: bool = True,
):
    result = partan_mns(
        fx=fx,
        x=x_start,
        max_iter=max_iter,
        eps=eps,
    )

    print("ПАРТАН-МНС")
    print(f"f(x) = {sp.expand(fx)}")
    for i, grad_i in enumerate(result["gradient_expr"], start=1):
        print(f"df/dx{i} = {grad_i}")

    for item in result["history"]:
        k = item["k"]
        method = item["method"]

        if method == "stop":
            print()
            print(f"Зупинка на ітерації {k + 1}:")
            print(f"x^({k}) = ({item['x'][0]:.3f}, {item['x'][1]:.3f})")
            print(f"||grad f(x^({k}))|| = {item['grad_norm_x']:.6f} <= {result['eps']}")
            break

        print()
        if method == "mns":
            print(f"Ітерація {k + 1} (МНС):")
        else:
            print(f"Ітерація {k + 1} (ПАРТАН):")

        print(f"x^({k}) = ({item['x'][0]:.3f}, {item['x'][1]:.3f})")
        print(f"f(x^({k})) = {item['f_x']:.3f}")
        print(f"grad f(x^({k})) = ({item['grad_x'][0]:.3f}, {item['grad_x'][1]:.3f})")
        print(f"||grad f(x^({k}))|| = {item['grad_norm_x']:.3f}")
        print(f"s^({k}) = ({item['s'][0]:.3f}, {item['s'][1]:.3f})")
        print(f"lambda_{k} = {item['lambda_opt']:.6f}")
        print(f"x^({k + 1}) = ({item['x_next'][0]:.3f}, {item['x_next'][1]:.3f})")
        print(f"f(x^({k + 1})) = {item['f_x_next']:.3f}")

    print()
    print(f"Фінальна точка: ({result['x_final'][0]:.6f}, {result['x_final'][1]:.6f})")
    print(f"f(x*) = {result['f_final']:.6f}")
    print(f"||grad f(x*)|| = {result['grad_norm_final']:.6f}")

    if show_plot:
        plot_partan_mns(result)
