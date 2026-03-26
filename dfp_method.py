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
    return float(lambda_opt), phi_expr, dphi_expr


def dfp_method(
    fx: sp.Expr,
    x: Sequence[float],
    max_iter: int,
    eps: float = 1e-5,
    a0: np.ndarray | None = None,
):
    """
    Метод Девідона-Флетчера-Пауелла (DFP):
      s_k = -A_k g_k
      x_{k+1} = x_k + lambda_k s_k
      A_{k+1} = A_k + (dx dx^T)/(dx^T dg) - (A_k dg dg^T A_k)/(dg^T A_k dg)
    """
    if max_iter <= 0:
        raise ValueError("max_iter >= 1")

    variables = tuple(sorted(fx.free_symbols, key=lambda s: s.name))
    xk = np.asarray(x, dtype=float).reshape(-1)
    n = xk.size

    if len(variables) != n:
        raise ValueError("Кількість змінних у fx має збігатися з розмірністю x.")

    if a0 is None:
        A_k = np.eye(n, dtype=float)
    else:
        A_k = np.asarray(a0, dtype=float)
        if A_k.shape != (n, n):
            raise ValueError("a0 має бути матрицею n x n.")
    A0 = A_k.copy()

    gradient_expr = [sp.diff(fx, var) for var in variables]
    hessian_expr = sp.Matrix(sp.hessian(fx, variables))
    f_func = sp.lambdify(variables, fx, modules="numpy")
    grad_func = sp.lambdify(variables, gradient_expr, modules="numpy")

    history = []
    points = [xk.copy()]

    for k in range(max_iter):
        grad_k = np.asarray(grad_func(*xk), dtype=float).reshape(-1)
        grad_norm_k = float(np.linalg.norm(grad_k))
        f_k = float(f_func(*xk))

        s_k = -A_k @ grad_k

        if grad_norm_k <= eps:
            history.append(
                {
                    "k": k,
                    "x": xk.copy(),
                    "f_x": f_k,
                    "grad_x": grad_k.copy(),
                    "grad_norm_x": grad_norm_k,
                    "A_k": A_k.copy(),
                    "s": s_k.copy(),
                    "lambda_opt": 0.0,
                    "x_next": xk.copy(),
                    "f_x_next": f_k,
                    "grad_x_next": grad_k.copy(),
                    "grad_norm_x_next": grad_norm_k,
                    "A_next": A_k.copy(),
                    "phi_expr": sp.Integer(0),
                    "dphi_expr": sp.Integer(0),
                }
            )
            break

        lambda_opt, phi_expr, dphi_expr = _compute_optimal_step(
            fx=fx, variables=variables, xk=xk, s_k=s_k
        )
        x_next = xk + lambda_opt * s_k

        grad_next = np.asarray(grad_func(*x_next), dtype=float).reshape(-1)
        grad_norm_next = float(np.linalg.norm(grad_next))
        f_next = float(f_func(*x_next))

        dx = x_next - xk
        dg = grad_next - grad_k

        dx_t_dg = float(dx @ dg)
        dg_t_A_dg = float(dg @ A_k @ dg)

        if abs(dx_t_dg) <= 1e-14 or abs(dg_t_A_dg) <= 1e-14:
            A_next = A_k.copy()
        else:
            term1 = np.outer(dx, dx) / dx_t_dg
            Adg = A_k @ dg
            term2 = np.outer(Adg, Adg) / dg_t_A_dg
            A_next = A_k + term1 - term2
            A_next = 0.5 * (A_next + A_next.T)

        history.append(
            {
                "k": k,
                "x": xk.copy(),
                "f_x": f_k,
                "grad_x": grad_k.copy(),
                "grad_norm_x": grad_norm_k,
                "A_k": A_k.copy(),
                "s": s_k.copy(),
                "lambda_opt": lambda_opt,
                "x_next": x_next.copy(),
                "f_x_next": f_next,
                "grad_x_next": grad_next.copy(),
                "grad_norm_x_next": grad_norm_next,
                "A_next": A_next.copy(),
                "phi_expr": phi_expr,
                "dphi_expr": dphi_expr,
            }
        )

        xk = x_next
        A_k = A_next
        points.append(xk.copy())

    grad_final = np.asarray(grad_func(*xk), dtype=float).reshape(-1)
    grad_norm_final = float(np.linalg.norm(grad_final))
    f_final = float(f_func(*xk))

    return {
        "fx": fx,
        "variables": variables,
        "gradient_expr": gradient_expr,
        "hessian_expr": hessian_expr,
        "A0": A0,
        "max_iter": int(max_iter),
        "eps": float(eps),
        "history": history,
        "points": np.vstack(points),
        "x_final": xk,
        "f_final": f_final,
        "grad_final": grad_final,
        "grad_norm_final": grad_norm_final,
    }


def plot_dfp_method(result: dict):
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
    ax.contour(X, Y, Z, levels=20, cmap="coolwarm", alpha=0.5)
    ax.plot(points[:, 0], points[:, 1], "o-", color="tab:blue", linewidth=2.0, markersize=5)

    for i, point in enumerate(points):
        ax.annotate(f"x^{i}", (point[0], point[1]), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Метод ДФП: траєкторія")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


def run_dfp_method(
    fx: sp.Expr,
    x_start: Sequence[float],
    max_iter: int,
    a0: np.ndarray | None = None,
    eps: float = 1e-6,
    show_plot: bool = True,
):
    result = dfp_method(
        fx=fx,
        x=x_start,
        max_iter=max_iter,
        a0=a0,
        eps=eps,
    )

    print("Метод ДФП")
    print(f"f(x) = {sp.expand(fx)}")
    for i, grad_i in enumerate(result["gradient_expr"], start=1):
        print(f"df/dx{i} = {grad_i}")
    print(f"A^(0) =\n{np.array2string(result['A0'], precision=3, suppress_small=True)}")

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
        print(f"||grad f(x^({k + 1}))|| = {item['grad_norm_x_next']:.3f}")
        print(f"A^({k}) =\n{np.array2string(item['A_k'], precision=3, suppress_small=True)}")
        print(f"A^({k + 1}) =\n{np.array2string(item['A_next'], precision=3, suppress_small=True)}")

    print()
    print(f"Фінальна точка: ({result['x_final'][0]:.6f}, {result['x_final'][1]:.6f})")
    print(f"f(x*) = {result['f_final']:.6f}")
    print(f"||grad f(x*)|| = {result['grad_norm_final']:.6f}")

    if len(result["history"]) >= 2:
        x1 = result["history"][0]["x_next"]
        x2 = result["history"][1]["x_next"]
        A2 = result["history"][1]["A_next"]

        hessian_expr = result["hessian_expr"]
        substitutions = {var: float(x2[i]) for i, var in enumerate(result["variables"])}
        hessian_num = np.asarray(hessian_expr.subs(substitutions), dtype=float)
        try:
            hessian_inv = np.linalg.inv(hessian_num)
        except np.linalg.LinAlgError:
            hessian_inv = np.linalg.pinv(hessian_num)

        print()
        print("Підсумок задачі (точність 3 знаки після коми):")
        print(f"x^(1) = ({x1[0]:.3f}, {x1[1]:.3f})")
        print(f"x^(2) = ({x2[0]:.3f}, {x2[1]:.3f})")
        print(f"A2 =\n{np.array2string(A2, precision=3, suppress_small=True)}")
        print(f"[H]^-1 =\n{np.array2string(hessian_inv, precision=3, suppress_small=True)}")
        diff_norm = float(np.linalg.norm(A2 - hessian_inv))
        print(f"||A2 - [H]^-1|| = {diff_norm:.3e}")

    if show_plot:
        plot_dfp_method(result)

    return result


if __name__ == "__main__":
    x1, x2 = sp.symbols("x1 x2")
    fx = 2 * x1**2 + x1 * x2 + x2**2
    run_dfp_method(
        fx=fx,
        x_start=(2.0, 2.0),
        max_iter=5,
    )
