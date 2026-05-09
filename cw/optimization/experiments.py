import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from partan_steepest_descent_method import partan_mns
    from steepest_descent_optimal_step import steepest_descent_optimal_step
    from optimization.functions import X_START, power_function
    from optimization.partan_steepest_descent import partan_steepest_descent
    from optimization.penalty import (
        circle_constraint,
        make_external_penalty_function,
        squared_penalty,
        total_violation,
    )
    from optimization.plots import plot_method_calls_comparison, plot_trajectory
    from optimization.steepest_descent import steepest_descent
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from partan_steepest_descent_method import partan_mns
    from steepest_descent_optimal_step import steepest_descent_optimal_step
    from .functions import X_START, power_function
    from .partan_steepest_descent import partan_steepest_descent
    from .penalty import (
        circle_constraint,
        make_external_penalty_function,
        squared_penalty,
        total_violation,
    )
    from .plots import plot_method_calls_comparison, plot_trajectory
    from .steepest_descent import steepest_descent


BASE_PARAMS = {
    "max_iter": 1000,
    "eps": 1e-6,
    "derivative_h": 1e-4,
    "gradient_scheme": "central",
    "line_search_method": "golden",
    "line_search_eps": 1e-12,
    "sven_alpha": 0.01,
    "stop_criterion": "combined",
}

EXPERIMENTS = {
    "derivative_h": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    "gradient_scheme": ["forward", "backward", "central"],
    "line_search_method": ["golden", "dsk_powell"],
    "line_search_eps": [1e-6, 1e-8, 1e-10, 1e-12],
    "sven_alpha": [0.001, 0.005, 0.01, 0.05, 0.1],
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
}


def format_point(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{v:.8f}" for v in arr) + "]"


def result_row(parameter_value, result):
    return {
        "parameter_value": parameter_value,
        "x_final": format_point(result["x_final"]),
        "f_final": float(result["f_final"]),
        "grad_norm_final": float(result["grad_norm_final"]),
        "iterations": int(result["iterations"]),
        "func_calls": int(result["func_calls"]),
        "status": STATUS_LABELS.get(result.get("status", "unknown"), result.get("status", "unknown")),
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
                "f_final": float(result["f_final"]),
                "grad_norm_final": float(result["grad_norm_final"]),
                "iterations": int(result["iterations"]),
                "func_calls": int(result["func_calls"]),
                "status": STATUS_LABELS.get(result.get("status", "unknown"), result.get("status", "unknown")),
            }
        )

    return pd.DataFrame(rows)


def symbolic_power_expr():
    x1, x2 = sp.symbols("x1 x2")
    return (10 * (x1 - x2) ** 2 + (x1 - 1) ** 2) ** 4


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


def _is_bar_plot(df):
    values = df["parameter_value"].to_list()
    return len(values) <= 3 or any(isinstance(value, str) for value in values)


def plot_calls_table(df, title, ax=None):
    own_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
        own_fig = True

    x = np.arange(len(df))
    if _is_bar_plot(df):
        ax.bar(x, df["func_calls"], color="tab:blue", width=0.62)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    else:
        ax.plot(x, df["func_calls"], "o-", color="tab:blue", linewidth=1.8)
        ax.grid(True, linestyle="--", alpha=0.35)
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
    r_values=(1, 10, 100, 1000, 10000),
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
                "r": r,
                "x_final": format_point(x_final),
                "f_original": float(power_function(x_final)),
                "F_penalty": float(result["f_final"]),
                "constraint_value": float(g_value),
                "violation": float(violation),
                "iterations": int(result["iterations"]),
                "func_calls": int(result["func_calls"]),
                "status": STATUS_LABELS.get(
                    result.get("status", "unknown"),
                    result.get("status", "unknown"),
                ),
            }
        )

        x_current = x_final

    return pd.DataFrame(rows)


def compare_penalty_methods(base_params=None, r_values=(1, 10, 100, 1000, 10000)):
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
    plot_trajectory(power_function, mns_res["points"], "Траєкторія МНС", cmap="Blues", ax=axes[0, 0])
    plot_trajectory(power_function, partan_res["points"], "Траєкторія ПАРТАН-МНС", cmap="Oranges", ax=axes[0, 1])
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
    result = run_experiments()
    print(f"Готово. Графіки збережено в: {result['output_dir']}")
    print(result["tables"]["methods_comparison"].rename(columns=DISPLAY_COLUMN_LABELS).to_string(index=False))
