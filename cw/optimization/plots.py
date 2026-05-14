import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {"МНС": "orange", "ПАРТАН-МНС": "navy"}


def method_color(method_name: str, default: str = "navy") -> str:
    return METHOD_COLORS.get(str(method_name), default)


def method_color_from_title(title: str, default: str = "navy") -> str:
    title_text = str(title)
    if "ПАРТАН-МНС" in title_text or "PARTAN-MNS" in title_text:
        return method_color("ПАРТАН-МНС", default)
    if "МНС" in title_text or "MNS" in title_text:
        return method_color("МНС", default)
    return default


def use_light_plot_theme():
    plt.style.use("default")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#eaeaf2"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.color"] = "#222222"
    plt.rcParams["axes.labelcolor"] = "#222222"
    plt.rcParams["axes.edgecolor"] = "#777777"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["navy", "orange", "tab:green", "tab:red", "tab:purple"])
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.color"] = "#222222"
    plt.rcParams["ytick.color"] = "#222222"
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["grid.color"] = "white"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.8
    plt.rcParams["grid.alpha"] = 0.85
    plt.rcParams["lines.linewidth"] = 1.7
    plt.rcParams["lines.markersize"] = 4.5
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "#d0d0d0"
    plt.rcParams["legend.framealpha"] = 0.85
    plt.rcParams["legend.fontsize"] = 8


use_light_plot_theme()


def _finite_values(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _axis_limits(values, log_scale: bool = False, pad_ratio: float = 0.08):
    finite = _finite_values(values)
    if finite.size == 0:
        return None

    if log_scale:
        finite = finite[finite > 0]
        if finite.size == 0:
            return None
        min_value = float(finite.min())
        max_value = float(finite.max())
        if np.isclose(min_value, max_value):
            return min_value / 10, max_value * 10
        padding = (max_value / min_value) ** pad_ratio
        return min_value / padding, max_value * padding

    min_value = float(finite.min())
    max_value = float(finite.max())
    if np.isclose(min_value, max_value):
        padding = max(abs(min_value) * 0.1, 1.0)
    else:
        padding = (max_value - min_value) * pad_ratio
    return min_value - padding, max_value + padding


def _apply_axis_limits(ax, x_values=None, y_values=None, x_log: bool = False, y_log: bool = False):
    if x_values is not None:
        x_limits = _axis_limits(x_values, log_scale=x_log)
        if x_limits is not None:
            ax.set_xlim(*x_limits)
    if y_values is not None:
        y_limits = _axis_limits(y_values, log_scale=y_log)
        if y_limits is not None:
            ax.set_ylim(*y_limits)


def _make_grid(points: np.ndarray, pad_ratio: float = 0.35, grid_size: int = 250):
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    x_span = max(float(x_vals.max() - x_vals.min()), 1.0)
    y_span = max(float(y_vals.max() - y_vals.min()), 1.0)
    span = max(x_span, y_span)
    pad = pad_ratio * span

    x_min = float(x_vals.min()) - pad
    x_max = float(x_vals.max()) + pad
    y_min = float(y_vals.min()) - pad
    y_max = float(y_vals.max()) + pad

    grid_x = np.linspace(x_min, x_max, grid_size)
    grid_y = np.linspace(y_min, y_max, grid_size)
    return np.meshgrid(grid_x, grid_y)


def plot_trajectory(
    f,
    points: np.ndarray,
    title: str,
    cmap: str = "viridis",
    levels: int = 25,
    trajectory_color: str | None = None,
    ax=None,
):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("plot_trajectory підтримує лише двовимірні точки.")

    X, Y = _make_grid(pts)
    Z = np.vectorize(lambda x1, x2: f(np.array([x1, x2], dtype=float)))(X, Y)

    own_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
        own_fig = True

    ax.contour(X, Y, Z, levels=levels, cmap=cmap, alpha=0.65)
    line_color = trajectory_color or method_color_from_title(title)
    ax.plot(pts[:, 0], pts[:, 1], "o-", color=line_color, linewidth=1.7, markersize=4.5)
    for i, point in enumerate(pts):
        ax.annotate(f"x{i}", (point[0], point[1]), textcoords="offset points", xytext=(4, 4), fontsize=8)

    if title:
        ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, linestyle="--", alpha=0.85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(X.min()), float(X.max()))
    ax.set_ylim(float(Y.min()), float(Y.max()))

    if own_fig:
        plt.tight_layout()
        plt.show()


def plot_calls_vs_param(rows, parameter_name: str, title: str, ax=None):
    values = [row["parameter_value"] for row in rows]
    calls = [row["func_calls"] for row in rows]

    own_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
        own_fig = True

    plot_color = method_color_from_title(title)
    ax.plot(values, calls, "o-", color=plot_color)
    _apply_axis_limits(ax, values, calls)
    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Кількість викликів функції")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.85)

    if own_fig:
        plt.tight_layout()
        plt.show()


def plot_method_calls_comparison(rows, title: str, ax=None):
    methods = [row["method"] for row in rows]
    calls = [row["func_calls"] for row in rows]

    own_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
        own_fig = True

    ax.bar(methods, calls, color=[method_color(method) for method in methods])
    _apply_axis_limits(ax, y_values=[0, *calls])
    ax.set_ylabel("Кількість викликів функції")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.85)

    if own_fig:
        plt.tight_layout()
        plt.show()


def _point_from_value(value):
    if isinstance(value, str):
        return np.fromstring(value.strip("[]"), sep=",")
    return np.asarray(value, dtype=float).reshape(-1)


def _is_numeric_series(values):
    try:
        np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return False
    return True


def plot_metric_by_parameter(
    table,
    parameter_col,
    metric_col,
    title,
    xlabel,
    ylabel,
    method_col="method",
    log_x=False,
    log_y=False,
    ax=None,
):
    own_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
        own_fig = True

    has_methods = method_col in table.columns
    groups = list(table.groupby(method_col, sort=False)) if has_methods else [(None, table)]
    values = table[parameter_col].to_list()
    numeric_x = _is_numeric_series(values)
    categorical = not numeric_x
    plotted_x = []
    plotted_y = []

    if categorical:
        labels = [str(value) for value in table[parameter_col].drop_duplicates().to_list()]
        x = np.arange(len(labels))
        group_count = len(groups)
        width = min(0.8 / max(group_count, 1), 0.34)
        offsets = (np.arange(group_count) - (group_count - 1) / 2) * width

        for offset, (method_name, group) in zip(offsets, groups):
            label_to_metric = dict(zip(group[parameter_col].astype(str), group[metric_col]))
            y = [label_to_metric.get(label, np.nan) for label in labels]
            ax.bar(x + offset, y, width=width, label=method_name, color=method_color(method_name))
            plotted_y.extend(y)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
    else:
        for method_name, group in groups:
            x = group[parameter_col].to_numpy(dtype=float)
            y = group[metric_col].to_numpy(dtype=float)
            ax.plot(x, y, "o-", linewidth=1.8, label=method_name, color=method_color(method_name))
            plotted_x.extend(x)
            plotted_y.extend(y)
        if log_x:
            ax.set_xscale("log")

    if log_y:
        finite_values = _finite_values(plotted_y)
        positive_values = finite_values[finite_values > 0]
        if len(finite_values) > 0 and len(positive_values) == len(finite_values):
            ax.set_yscale("log")
            _apply_axis_limits(ax, plotted_x if not categorical else None, plotted_y, x_log=log_x, y_log=True)
        else:
            ax.set_yscale("symlog", linthresh=1e-12)
            _apply_axis_limits(ax, plotted_x if not categorical else None, plotted_y, x_log=log_x, y_log=False)
    else:
        y_values_for_limits = [0, *plotted_y] if categorical else plotted_y
        _apply_axis_limits(ax, plotted_x if not categorical else None, y_values_for_limits, x_log=log_x, y_log=False)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", alpha=0.85)
    if has_methods:
        ax.legend()

    if own_fig:
        plt.tight_layout()
        plt.show()

    return ax


def plot_circle_constraint(ax):
    theta = np.linspace(0, 2 * np.pi, 500)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.fill(x, y, color="tab:green", alpha=0.08, label="допустима область")
    ax.plot(x, y, color="black", linewidth=1.4, label="межа x1^2 + x2^2 = 1")
    return ax


def plot_penalty_trajectory(table, title, ax=None):
    own_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
        own_fig = True

    plot_circle_constraint(ax)
    points = np.vstack([_point_from_value(value) for value in table["x_final"]])
    plot_color = method_color_from_title(title)
    ax.plot(points[:, 0], points[:, 1], "o-", color=plot_color, linewidth=1.8, label="точки після r")
    for r_value, point in zip(table["r"], points):
        ax.annotate(f"r={r_value:g}", point, textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.scatter([1.0], [1.0], marker="x", s=90, color="tab:red", linewidths=2.2, label="безумовний мінімум")
    if "x_start" in table.columns:
        start = _point_from_value(table.iloc[0]["x_start"])
        ax.scatter([start[0]], [start[1]], marker="s", s=55, color="tab:purple", label="початкова точка")
        points_for_limits = np.vstack([points, start])
    else:
        points_for_limits = points
    final = points[-1]
    ax.scatter([final[0]], [final[1]], s=95, color=plot_color, edgecolor="black", linewidth=1.0, label="фінальна точка")
    if title:
        ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    _apply_axis_limits(ax, points_for_limits[:, 0], points_for_limits[:, 1])
    ax.grid(True, linestyle="--", alpha=0.85)
    ax.legend(fontsize=8)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return ax


def plot_convergence_by_iteration(results: dict, title: str = "Збіжність значення функції для МНС і ПАРТАН-МНС", ax=None):
    own_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
        own_fig = True

    for method_name, result in results.items():
        history = result["history"]
        iterations = [h["k"] for h in history]
        f_values = [h["f_x"] for h in history]
        ax.plot(iterations, f_values, "o-", linewidth=1.5, markersize=3,
                label=method_name, color=method_color(method_name))

    all_iterations = [h["k"] for result in results.values() for h in result["history"]]
    all_f_values = [h["f_x"] for result in results.values() for h in result["history"]]
    if _finite_values(all_f_values).size > 0 and np.all(_finite_values(all_f_values) > 0):
        ax.set_yscale("log")
        _apply_axis_limits(ax, all_iterations, all_f_values, y_log=True)
    else:
        ax.set_yscale("symlog", linthresh=1e-20)
        _apply_axis_limits(ax, all_iterations, all_f_values)
    ax.set_xlabel("Номер ітерації k")
    ax.set_ylabel("f(x_k)")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.85)
    ax.legend()

    if own_fig:
        plt.tight_layout()
        plt.show()
    return ax


def plot_gradient_norm_by_iteration(results: dict, title: str = "Зміна норми градієнта в процесі мінімізації", ax=None):
    own_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
        own_fig = True

    for method_name, result in results.items():
        history = result["history"]
        iterations = [h["k"] for h in history]
        grad_norms = [h["grad_norm"] for h in history]
        ax.plot(iterations, grad_norms, "o-", linewidth=1.5, markersize=3,
                label=method_name, color=method_color(method_name))

    all_iterations = [h["k"] for result in results.values() for h in result["history"]]
    all_grad_norms = [h["grad_norm"] for result in results.values() for h in result["history"]]
    if _finite_values(all_grad_norms).size > 0 and np.all(_finite_values(all_grad_norms) > 0):
        ax.set_yscale("log")
        _apply_axis_limits(ax, all_iterations, all_grad_norms, y_log=True)
    else:
        ax.set_yscale("symlog", linthresh=1e-20)
        _apply_axis_limits(ax, all_iterations, all_grad_norms)
    ax.set_xlabel("Номер ітерації k")
    ax.set_ylabel("||∇f(x_k)||")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.85)
    ax.legend()

    if own_fig:
        plt.tight_layout()
        plt.show()
    return ax


def plot_cumulative_calls_by_iteration(results: dict, title: str = "Накопичена кількість викликів функції під час мінімізації", ax=None):
    own_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
        own_fig = True

    for method_name, result in results.items():
        history = result["history"]
        iterations = [h["k"] for h in history]
        cumulative = [h["func_calls"] for h in history]
        ax.plot(iterations, cumulative, "o-", linewidth=1.5, markersize=3,
                label=method_name, color=method_color(method_name))

    all_iterations = [h["k"] for result in results.values() for h in result["history"]]
    all_cumulative = [h["func_calls"] for result in results.values() for h in result["history"]]
    _apply_axis_limits(ax, all_iterations, all_cumulative)
    ax.set_xlabel("Номер ітерації k")
    ax.set_ylabel("Накопичена кількість викликів функції")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.85)
    ax.legend()

    if own_fig:
        plt.tight_layout()
        plt.show()
    return ax


def plot_final_comparison(table, metric_col, title, ax=None):
    own_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
        own_fig = True

    values = table[metric_col].to_numpy(dtype=float)
    methods = table["method"].astype(str)
    ax.bar(methods, values, color=[method_color(method) for method in methods])
    _apply_axis_limits(ax, y_values=np.concatenate(([0.0], values)))
    if title:
        ax.set_title(title)
    ax.set_xlabel("метод")
    ax.set_ylabel(metric_col)
    ax.grid(True, axis="y", linestyle="--", alpha=0.85)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return ax
