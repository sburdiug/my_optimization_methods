import numpy as np

from .functions import FunctionCounter
from .gradients import numerical_gradient
from .line_search import line_search, sven_delta, sven_interval


def partan_steepest_descent(
    f,
    x_start,
    max_iter: int = 1000,
    eps: float = 1e-4,
    derivative_h: float = 1e-4,
    gradient_scheme: str = "central",
    line_search_method: str = "golden",
    line_search_eps: float = 1e-3,
    sven_alpha: float = 0.01,
    stop_criterion: str = "combined",
):
    if max_iter <= 0:
        raise ValueError("max_iter має бути додатним.")
    if eps <= 0:
        raise ValueError("eps має бути додатним.")
    if sven_alpha <= 0:
        raise ValueError("sven_alpha має бути додатним.")
    if stop_criterion not in {"gradient", "combined"}:
        raise ValueError("stop_criterion має бути 'gradient' або 'combined'.")

    f_counted = FunctionCounter(f)
    xk = np.asarray(x_start, dtype=float).reshape(-1)
    history = []
    points = [xk.copy()]
    status = "max_iter"

    for k in range(max_iter):
        grad_k = numerical_gradient(f_counted, xk, h=derivative_h, scheme=gradient_scheme)
        grad_norm_k = float(np.linalg.norm(grad_k))
        f_k = float(f_counted(xk))

        if not np.isfinite(grad_norm_k) or not np.isfinite(f_k):
            history.append(
                {
                    "k": k,
                    "method_name": "stop",
                    "x": xk.copy(),
                    "f_x": f_k,
                    "grad": grad_k.copy(),
                    "grad_norm": grad_norm_k,
                    "s": np.zeros_like(xk),
                    "lambda_opt": 0.0,
                    "x_next": xk.copy(),
                    "f_next": f_k,
                    "func_calls": int(f_counted.calls),
                }
            )
            status = "numerical_issue"
            points.append(xk.copy())
            break

        if stop_criterion == "gradient" and grad_norm_k <= eps:
            history.append(
                {
                    "k": k,
                    "method_name": "stop",
                    "x": xk.copy(),
                    "f_x": f_k,
                    "grad": grad_k.copy(),
                    "grad_norm": grad_norm_k,
                    "s": np.zeros_like(xk),
                    "lambda_opt": 0.0,
                    "x_next": xk.copy(),
                    "f_next": f_k,
                    "func_calls": int(f_counted.calls),
                }
            )
            status = "converged"
            points.append(xk.copy())
            break

        if k < 2:
            method_name = "mns"
            s_k = -grad_k
        else:
            if k % 2 == 0:
                method_name = "partan"
                s_k = xk - points[-3]
            else:
                method_name = "mns"
                s_k = -grad_k

        if np.linalg.norm(s_k) <= 1e-14:
            method_name = "mns"
            s_k = -grad_k

        lambda_opt = 0.0
        delta_k = 0.0
        x_next = xk.copy()
        f_next = f_k

        if np.linalg.norm(s_k) > 1e-14:
            phi = lambda lam: f_counted(xk + float(lam) * s_k)
            delta_k = sven_delta(xk, s_k, sven_alpha)
            a, b = sven_interval(phi, delta=delta_k)
            lambda_opt = line_search(
                phi,
                method=line_search_method,
                a=a,
                b=b,
                eps=line_search_eps,
            )
            x_next = xk + lambda_opt * s_k
            f_next = float(f_counted(x_next))
            if not np.isfinite(lambda_opt) or not np.isfinite(f_next) or not np.all(np.isfinite(x_next)):
                status = "numerical_issue"

        if stop_criterion == "combined" and status != "numerical_issue":
            denom = max(float(np.linalg.norm(xk)), 1e-12)
            rel_x = float(np.linalg.norm(x_next - xk)) / denom
            diff_f = abs(f_next - f_k)
            if np.isfinite(rel_x) and np.isfinite(diff_f) and rel_x <= eps and diff_f <= eps:
                status = "converged"

        history.append(
            {
                "k": k,
                "method_name": method_name,
                "x": xk.copy(),
                "f_x": f_k,
                "grad": grad_k.copy(),
                "grad_norm": grad_norm_k,
                "s": s_k.copy(),
                "sven_delta": float(delta_k),
                "lambda_opt": float(lambda_opt),
                "x_next": x_next.copy(),
                "f_next": float(f_next),
                "func_calls": int(f_counted.calls),
            }
        )

        xk = x_next
        points.append(xk.copy())
        if status in {"converged", "numerical_issue"}:
            break

    grad_final = numerical_gradient(f_counted, xk, h=derivative_h, scheme=gradient_scheme)
    grad_norm_final = float(np.linalg.norm(grad_final))
    f_final = float(f_counted(xk))

    return {
        "method": "partan_steepest_descent",
        "x_final": xk,
        "f_final": f_final,
        "grad_final": grad_final,
        "grad_norm_final": grad_norm_final,
        "iterations": len(history),
        "func_calls": int(f_counted.calls),
        "points": np.vstack(points),
        "history": history,
        "status": status,
        "params": {
            "max_iter": max_iter,
            "eps": eps,
            "derivative_h": derivative_h,
            "gradient_scheme": gradient_scheme,
            "line_search_method": line_search_method,
            "line_search_eps": line_search_eps,
            "sven_alpha": sven_alpha,
            "stop_criterion": stop_criterion,
        },
    }
