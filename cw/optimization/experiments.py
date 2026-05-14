import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from partan_steepest_descent_method import partan_mns
    from steepest_descent_optimal_step import steepest_descent_optimal_step
    from optimization.functions import X_MIN, X_START, power_function
    from optimization.line_search import line_search, sven_delta, sven_interval
    from optimization.partan_steepest_descent import partan_steepest_descent
    from optimization.penalty import (
        circle_constraint,
        make_external_penalty_function,
        total_violation,
    )
    from optimization.plots import method_color_from_title, plot_method_calls_comparison, plot_trajectory
    from optimization.steepest_descent import steepest_descent
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from partan_steepest_descent_method import partan_mns
    from steepest_descent_optimal_step import steepest_descent_optimal_step
    from .functions import X_MIN, X_START, power_function
    from .line_search import line_search, sven_delta, sven_interval
    from .partan_steepest_descent import partan_steepest_descent
    from .penalty import (
        circle_constraint,
        make_external_penalty_function,
        total_violation,
    )
    from .plots import method_color_from_title, plot_method_calls_comparison, plot_trajectory
    from .steepest_descent import steepest_descent


BASE_PARAMS = {
    "max_iter": 1000,
    "eps": 1e-3,
    "derivative_h": 1e-3,
    "gradient_scheme": "central",
    "line_search_method": "dsk_powell",
    "line_search_eps": 1e-8,
    "sven_alpha": 0.01,
    "stop_criterion": "combined",
}

EXPERIMENTS = {
    "derivative_h": [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10],
    "gradient_scheme": ["forward", "backward", "central"],
    "line_search_method": ["golden", "dsk_powell"],
    "line_search_eps": [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
    "sven_alpha": [1e-8, 1e-6, 1e-5, 1e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0],
    "stop_criterion": ["gradient", "combined"],
}

METHODS = {
    "МНС": steepest_descent,
    "ПАРТАН-МНС": partan_steepest_descent,
}

STATUS_LABELS = {
    "converged": "збіжність",
    "max_iter": "ліміт ітерацій",
    "numerical_issue": "числова помилка",
    "unknown": "невідомо",
}

DISPLAY_COLUMN_LABELS = {
    "method": "метод",
    "parameter_value": "значення параметра",
    "x_final": "кінцева точка",
    "f_final": "кінцеве значення функції",
    "grad_norm_final": "норма градієнта",
    "iterations": "кількість ітерацій",
    "func_calls": "кількість викликів функції",
    "status": "статус",
    "r": "коефіцієнт штрафу r",
    "f_original": "значення початкової функції",
    "F_penalty": "значення штрафної функції",
    "constraint_value": "значення обмеження g(x)",
    "violation": "порушення обмеження",
    "distance_to_boundary": "відстань до межі області",
    "r_final": "фінальний коефіцієнт штрафу r",
    "x_start": "початкова точка",
    "iterations_total": "сумарна кількість ітерацій",
    "func_calls_total": "сумарна кількість викликів функції",
    "x_error": "відстань до точного мінімуму",
}


def status_label(status: object) -> str:
    status_key = str(status)
    return STATUS_LABELS.get(status_key, status_key)


def result_float(result: Mapping[str, Any], key: str) -> float:
    return float(result[key])


def result_int(result: Mapping[str, Any], key: str) -> int:
    return int(result[key])


def format_point(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{v:.8f}" for v in arr) + "]"


def distance_to_circle_boundary(x):
    x = np.asarray(x, dtype=float)
    return abs(float(np.linalg.norm(x)) - 1.0)


def method_display_name(method_fn):
    method_name = getattr(method_fn, "__name__", "")
    if method_name == "steepest_descent":
        return "МНС"
    if method_name == "partan_steepest_descent":
        return "ПАРТАН-МНС"
    return method_name or "метод"


def result_row(parameter_value, result):
    return {
        "parameter_value": parameter_value,
        "x_final": format_point(result["x_final"]),
        "f_final": result_float(result, "f_final"),
        "grad_norm_final": result_float(result, "grad_norm_final"),
        "x_error": float(np.linalg.norm(np.asarray(result["x_final"], dtype=float).reshape(-1) - X_MIN)),
        "iterations": result_int(result, "iterations"),
        "func_calls": result_int(result, "func_calls"),
        "status": status_label(result.get("status", "unknown")),
    }


def sweep(method_fn, parameter_name, values, base_params=None):
    base_params = dict(BASE_PARAMS if base_params is None else base_params)
    rows = []

    for value in values:
        params = dict(base_params)
        params[parameter_name] = value
        result = method_fn(power_function, X_START, **params)
        rows.append(result_row(value, result))

    return pd.DataFrame(rows)


def compare_methods(base_params=None):
    base_params = dict(BASE_PARAMS if base_params is None else base_params)
    rows = []

    for method_name, method_fn in METHODS.items():
        result = method_fn(power_function, X_START, **base_params)
        rows.append(
            {
                "method": method_name,
                "x_final": format_point(result["x_final"]),
                "f_final": result_float(result, "f_final"),
                "grad_norm_final": result_float(result, "grad_norm_final"),
                "iterations": result_int(result, "iterations"),
                "func_calls": result_int(result, "func_calls"),
                "status": status_label(result.get("status", "unknown")),
            }
        )

    return pd.DataFrame(rows)


def compare_line_search_precision(
    base_params=None,
    eps_values=None,
    line_search_variants=None,
):
    params_template = dict(BASE_PARAMS if base_params is None else base_params)
    eps_values = list(
        eps_values
        if eps_values is not None
        else [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    )
    line_search_variants = dict(
        line_search_variants
        if line_search_variants is not None
        else {"ДСК-Пауелла": "dsk_powell", "Golden ratio": "golden"}
    )

    results_by_line_search = {}
    for display_name, method_key in line_search_variants.items():
        rows = []
        for eps in eps_values:
            params = dict(params_template)
            params["line_search_eps"] = eps
            params["line_search_method"] = method_key
            result = partan_steepest_descent(power_function, X_START, **params)
            rows.append({"eps": eps, "func_calls": result["func_calls"]})
        results_by_line_search[display_name] = rows

    return results_by_line_search


def symbolic_power_expr():
    x1, x2 = sp.symbols("x1 x2")
    return (10 * (x1 - x2) ** 2 + (x1 - 1) ** 2) ** 4


def symbolic_external_penalty_expr(r=1):
    x1, x2 = sp.symbols("x1 x2")
    g = x1**2 + x2**2 - 1
    penalty = sp.Piecewise((g**2, g > 0), (0, True))
    return symbolic_power_expr() + float(r) * penalty


def points_from_old_mns(result):
    history = result["history"]
    if not history:
        return np.empty((0, 2), dtype=float)
    points = [np.asarray(history[0]["x"], dtype=float)]
    points.extend(np.asarray(item["x_next"], dtype=float) for item in history)
    return np.vstack(points)


def old_sympy_check(max_iter_mns=3, max_iter_partan=4, eps=1e-6):
    fx = symbolic_power_expr()
    old_mns = steepest_descent_optimal_step(fx=fx, x=X_START, max_iter=max_iter_mns, eps=eps)
    old_partan = partan_mns(fx=fx, x=X_START, max_iter=max_iter_partan, eps=eps)

    rows = [
        {
            "method": "МНС SymPy",
            "x_final": format_point(old_mns["x_final"]),
            "f_final": float(old_mns["f_final"]),
            "grad_norm_final": float(old_mns["grad_norm_final"]),
            "iterations": len(old_mns["history"]),
            "func_calls": "-",
            "status": "контроль",
        },
        {
            "method": "ПАРТАН-МНС SymPy",
            "x_final": format_point(old_partan["x_final"]),
            "f_final": float(old_partan["f_final"]),
            "grad_norm_final": float(old_partan["grad_norm_final"]),
            "iterations": len(old_partan["history"]),
            "func_calls": "-",
            "status": "контроль",
        },
    ]

    return {
        "table": pd.DataFrame(rows),
        "mns": old_mns,
        "partan": old_partan,
        "mns_points": points_from_old_mns(old_mns),
        "partan_points": np.asarray(old_partan["points"], dtype=float),
    }


def sympy_gradient_steepest_descent_penalty(
    x_start=(-1.0, -1.0),
    r=1,
    base_params=None,
):
    """
    Контроль МНС для штрафної функції з аналітичним градієнтом SymPy.

    Крок шукається тими самими методами Свена та золотого перерізу, але
    градієнт береться не чисельно, а з похідних SymPy для Piecewise-штрафу.
    """
    params = dict(BASE_PARAMS if base_params is None else base_params)
    x1, x2 = sp.symbols("x1 x2")
    variables = (x1, x2)
    fx = symbolic_external_penalty_expr(r)
    gradient_expr = [sp.diff(fx, var) for var in variables]
    f_func = sp.lambdify(variables, fx, modules="numpy")
    grad_func = sp.lambdify(variables, gradient_expr, modules="numpy")

    xk = np.asarray(x_start, dtype=float).reshape(-1)
    points = [xk.copy()]
    history = []
    func_calls = 0
    status = "max_iter"

    def counted_f(x):
        nonlocal func_calls
        func_calls += 1
        return float(f_func(float(x[0]), float(x[1])))

    for k in range(int(params["max_iter"])):
        grad_k = np.asarray(grad_func(float(xk[0]), float(xk[1])), dtype=float).reshape(-1)
        grad_norm_k = float(np.linalg.norm(grad_k))
        f_k = counted_f(xk)
        s_k = -grad_k
        delta_k = 0.0
        lambda_opt = 0.0
        x_next = xk.copy()
        f_next = f_k

        if not np.isfinite(grad_norm_k) or not np.isfinite(f_k):
            status = "numerical_issue"
        elif str(params["stop_criterion"]) == "gradient" and grad_norm_k <= float(params["eps"]):
            status = "converged"
        else:
            if np.linalg.norm(s_k) > 1e-14:
                phi = lambda lam: counted_f(xk + float(lam) * s_k)
                delta_k = sven_delta(xk, s_k, float(params["sven_alpha"]))
                a, b = sven_interval(phi, delta=delta_k)
                lambda_opt = line_search(
                    phi,
                    method=str(params["line_search_method"]),
                    a=a,
                    b=b,
                    eps=float(params["line_search_eps"]),
                )
                x_next = xk + lambda_opt * s_k
                f_next = counted_f(x_next)
                if not np.isfinite(lambda_opt) or not np.isfinite(f_next) or not np.all(np.isfinite(x_next)):
                    status = "numerical_issue"

            if str(params["stop_criterion"]) == "combined" and status != "numerical_issue":
                denom = max(float(np.linalg.norm(xk)), 1e-12)
                rel_x = float(np.linalg.norm(x_next - xk)) / denom
                diff_f = abs(f_next - f_k)
                if np.isfinite(rel_x) and np.isfinite(diff_f) and rel_x <= float(params["eps"]) and diff_f <= float(params["eps"]):
                    status = "converged"

        history.append(
            {
                "k": k,
                "x": xk.copy(),
                "f_x": f_k,
                "grad": grad_k.copy(),
                "grad_norm": grad_norm_k,
                "s": s_k.copy(),
                "sven_delta": float(delta_k),
                "lambda_opt": float(lambda_opt),
                "x_next": x_next.copy(),
                "f_next": float(f_next),
                "func_calls": int(func_calls),
            }
        )

        xk = x_next
        points.append(xk.copy())
        if status in {"converged", "numerical_issue"}:
            break

    grad_final = np.asarray(grad_func(float(xk[0]), float(xk[1])), dtype=float).reshape(-1)
    f_final = counted_f(xk)

    return {
        "method": "sympy_gradient_steepest_descent_penalty",
        "x_final": xk,
        "f_final": float(f_final),
        "grad_final": grad_final,
        "grad_norm_final": float(np.linalg.norm(grad_final)),
        "iterations": len(history),
        "func_calls": int(func_calls),
        "points": np.vstack(points),
        "history": history,
        "status": status,
        "fx": fx,
        "gradient_expr": gradient_expr,
    }


def sympy_penalty_s4_check(base_params=None):
    params = dict(BASE_PARAMS if base_params is None else base_params)
    x_start = (-1.0, -1.0)
    r = 1
    sympy_result = sympy_gradient_steepest_descent_penalty(
        x_start=x_start,
        r=r,
        base_params=params,
    )
    numpy_result = steepest_descent(
        make_external_penalty_function(power_function, [circle_constraint], r),
        x_start,
        **params,
    )

    rows = [
        {
            "method": "МНС NumPy",
            "x_start": format_point(x_start),
            "r": r,
            "x_final": format_point(numpy_result["x_final"]),
            "F_penalty": result_float(numpy_result, "f_final"),
            "grad_norm_final": result_float(numpy_result, "grad_norm_final"),
            "iterations": result_int(numpy_result, "iterations"),
            "func_calls": result_int(numpy_result, "func_calls"),
            "status": status_label(numpy_result.get("status", "unknown")),
        },
        {
            "method": "МНС SymPy-gradient",
            "x_start": format_point(x_start),
            "r": r,
            "x_final": format_point(sympy_result["x_final"]),
            "F_penalty": result_float(sympy_result, "f_final"),
            "grad_norm_final": result_float(sympy_result, "grad_norm_final"),
            "iterations": result_int(sympy_result, "iterations"),
            "func_calls": result_int(sympy_result, "func_calls"),
            "status": status_label(sympy_result.get("status", "unknown")),
        },
    ]

    return {
        "table": pd.DataFrame(rows),
        "numpy": numpy_result,
        "sympy": sympy_result,
    }


def _is_bar_plot(df):
    values = df["parameter_value"].to_list()
    return len(values) <= 3 or any(isinstance(value, str) for value in values)


def plot_calls_table(df, title, ax=None):
    own_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
        own_fig = True

    plot_color = method_color_from_title(title)
    x = np.arange(len(df))
    if _is_bar_plot(df):
        ax.bar(x, df["func_calls"], color=plot_color, width=0.62)
        ax.grid(True, axis="y", linestyle="--", alpha=0.85)
    else:
        ax.plot(x, df["func_calls"], "o-", color=plot_color, linewidth=1.8)
        ax.grid(True, linestyle="--", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["parameter_value"].astype(str), rotation=20, ha="right")
    ax.set_xlabel("Значення параметра")
    ax.set_ylabel("Кількість викликів функції")
    ax.set_title(title)

    if own_fig:
        plt.tight_layout()
        plt.show()


def plot_comparison_table(df, title, ax=None):
    own_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
        own_fig = True

    rows = df.to_dict("records")
    plot_method_calls_comparison(rows, title, ax=ax)

    if own_fig:
        plt.tight_layout()
        plt.show()


def run_all_experiments(base_params=None):
    base_params = dict(BASE_PARAMS if base_params is None else base_params)
    tables = {}

    for method_name, method_fn in METHODS.items():
        for parameter_name, values in EXPERIMENTS.items():
            tables[f"{method_name}_{parameter_name}"] = sweep(
                method_fn=method_fn,
                parameter_name=parameter_name,
                values=values,
                base_params=base_params,
            )

    tables["methods_comparison"] = compare_methods(base_params)
    return tables


def penalty_experiment(
    method_fn,
    base_params=None,
    r_values=(1, 10, 100, 1000, 10000, 100000),
    x_start=X_START,
):
    """
    Експеримент для методу штрафних функцій.

    На кожному етапі мінімізується F(x, r) = f(x) + r * P(x).
    Наступний запуск стартує з точки, знайденої на попередньому r.
    """
    base_params = dict(BASE_PARAMS if base_params is None else base_params)
    constraints = [circle_constraint]
    rows = []
    x_current = np.asarray(x_start, dtype=float).reshape(-1)

    for r in r_values:
        penalty_f = make_external_penalty_function(
            f=power_function,
            constraints=constraints,
            r=r,
        )

        result = method_fn(
            penalty_f,
            x_current,
            **base_params,
        )

        x_final = np.asarray(result["x_final"], dtype=float).reshape(-1)
        g_value = circle_constraint(x_final)
        violation = total_violation(x_final, constraints)

        rows.append(
            {
                "x_start": format_point(x_start),
                "r": r,
                "x_final": format_point(x_final),
                "f_original": float(power_function(x_final)),
                "F_penalty": result_float(result, "f_final"),
                "constraint_value": float(g_value),
                "violation": float(violation),
                "distance_to_boundary": float(distance_to_circle_boundary(x_final)),
                "iterations": result_int(result, "iterations"),
                "func_calls": result_int(result, "func_calls"),
                "status": status_label(result.get("status", "unknown")),
            }
        )

        x_current = x_final

    return pd.DataFrame(rows)


def penalty_experiment_summary(table, method=None):
    if table.empty:
        raise ValueError("Таблиця штрафного експерименту порожня.")

    final = table.iloc[-1]
    statuses = table["status"].astype(str).unique()
    status = final["status"] if len(statuses) == 1 else ", ".join(statuses)
    row = {
        "method": method,
        "r_final": final["r"],
        "x_final": final["x_final"],
        "f_original": final["f_original"],
        "F_penalty": final["F_penalty"],
        "constraint_value": final["constraint_value"],
        "violation": final["violation"],
        "distance_to_boundary": final["distance_to_boundary"],
        "iterations_total": int(table["iterations"].sum()),
        "func_calls_total": int(table["func_calls"].sum()),
        "status": status,
    }
    if method is None:
        row.pop("method")
    return row


def penalty_start_point_experiment(
    method_fn,
    start_points,
    base_params=None,
    r_values=(1, 10, 100, 1000, 10000),
):
    base_params = dict(BASE_PARAMS if base_params is None else base_params)
    rows = []

    for x_start in start_points:
        table = penalty_experiment(
            method_fn=method_fn,
            base_params=base_params,
            r_values=r_values,
            x_start=x_start,
        )
        summary = penalty_experiment_summary(table)
        summary["x_start"] = format_point(x_start)
        rows.append(
            {
                "x_start": summary["x_start"],
                "x_final": summary["x_final"],
                "f_original": summary["f_original"],
                "constraint_value": summary["constraint_value"],
                "violation": summary["violation"],
                "distance_to_boundary": summary["distance_to_boundary"],
                "iterations_total": summary["iterations_total"],
                "func_calls_total": summary["func_calls_total"],
                "status": summary["status"],
            }
        )

    return pd.DataFrame(rows)


def compare_penalty_start_points(
    base_params=None,
    start_points=((-1.5, 0.0), (1.5, 0.0), (0.0, 1.5), (-1.0, -1.0), (2.0, 2.0)),
    r_values=(1, 10, 100, 1000, 10000),
):
    base_params = dict(BASE_PARAMS if base_params is None else base_params)
    frames = []

    for method_name, method_fn in METHODS.items():
        table = penalty_start_point_experiment(
            method_fn=method_fn,
            start_points=start_points,
            base_params=base_params,
            r_values=r_values,
        )
        table.insert(0, "method", method_name)
        frames.append(table)

    return pd.concat(frames, ignore_index=True)


def compare_penalty_methods(base_params=None, r_values=(1, 10, 100, 1000, 10000, 100000)):
    base_params = dict(BASE_PARAMS if base_params is None else base_params)

    return {
        "МНС": penalty_experiment(
            method_fn=steepest_descent,
            base_params=base_params,
            r_values=r_values,
        ),
        "ПАРТАН-МНС": penalty_experiment(
            method_fn=partan_steepest_descent,
            base_params=base_params,
            r_values=r_values,
        ),
    }


def run_experiments(output_dir="results"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = run_all_experiments()

    mns_res = steepest_descent(power_function, X_START, **BASE_PARAMS)
    partan_res = partan_steepest_descent(power_function, X_START, **BASE_PARAMS)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_trajectory(
        power_function,
        mns_res["points"],
        "Траєкторія МНС",
        cmap="Oranges",
        trajectory_color="orange",
        ax=axes[0, 0],
    )
    plot_trajectory(
        power_function,
        partan_res["points"],
        "Траєкторія ПАРТАН-МНС",
        cmap="Blues",
        trajectory_color="navy",
        ax=axes[0, 1],
    )
    plot_calls_table(tables["МНС_derivative_h"], "Кількість викликів функції залежно від h (МНС)", ax=axes[1, 0])
    plot_comparison_table(tables["methods_comparison"], "Порівняння МНС і ПАРТАН-МНС за викликами функції", ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig(out / "summary_plots.png", dpi=180)
    plt.close(fig)

    penalty_tables = compare_penalty_methods(base_params=BASE_PARAMS)
    tables["penalty_МНС"] = penalty_tables["МНС"]
    tables["penalty_ПАРТАН-МНС"] = penalty_tables["ПАРТАН-МНС"]

    return {
        "output_dir": str(out.resolve()),
        "tables": tables,
    }


if __name__ == "__main__":
    run_result = run_experiments()
    print(f"Готово. Графіки збережено в: {run_result['output_dir']}")
    print(run_result["tables"]["methods_comparison"].rename(columns=DISPLAY_COLUMN_LABELS).to_string(index=False))
