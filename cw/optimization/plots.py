import matplotlib.pyplot as plt
import numpy as np


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
    ax.plot(pts[:, 0], pts[:, 1], "o-", color="tab:red", linewidth=1.7, markersize=4.5)
    for i, point in enumerate(pts):
        ax.annotate(f"x{i}", (point[0], point[1]), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_aspect("equal", adjustable="box")

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

    ax.plot(values, calls, "o-", color="tab:blue")
    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Кількість викликів функції")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

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

    ax.bar(methods, calls, color=["tab:green", "tab:orange"])
    ax.set_ylabel("Кількість викликів функції")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

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

    if categorical:
        labels = [str(value) for value in table[parameter_col].drop_duplicates().to_list()]
        x = np.arange(len(labels))
        group_count = len(groups)
        width = min(0.8 / max(group_count, 1), 0.34)
        offsets = (np.arange(group_count) - (group_count - 1) / 2) * width

        for offset, (method_name, group) in zip(offsets, groups):
            label_to_metric = dict(zip(group[parameter_col].astype(str), group[metric_col]))
            y = [label_to_metric.get(label, np.nan) for label in labels]
            ax.bar(x + offset, y, width=width, label=method_name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
    else:
        for method_name, group in groups:
            x = group[parameter_col].to_numpy(dtype=float)
            y = group[metric_col].to_numpy(dtype=float)
            ax.plot(x, y, "o-", linewidth=1.8, label=method_name)
        if log_x:
            ax.set_xscale("log")

    if log_y:
        ax.set_yscale("symlog", linthresh=1e-12)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
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
    ax.plot(points[:, 0], points[:, 1], "o-", color="tab:blue", linewidth=1.8, label="точки після r")
    for r_value, point in zip(table["r"], points):
        ax.annotate(f"r={r_value:g}", point, textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.scatter([1.0], [1.0], marker="x", s=90, color="tab:red", linewidths=2.2, label="безумовний мінімум")
    if "x_start" in table.columns:
        start = _point_from_value(table.iloc[0]["x_start"])
        ax.scatter([start[0]], [start[1]], marker="s", s=55, color="tab:purple", label="початкова точка")
    final = points[-1]
    ax.scatter([final[0]], [final[1]], s=95, color="tab:blue", edgecolor="black", linewidth=1.0, label="фінальна точка")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return ax


def plot_final_comparison(table, metric_col, title, ax=None):
    own_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
        own_fig = True

    ax.bar(table["method"].astype(str), table[metric_col].to_numpy(dtype=float), color=["tab:blue", "tab:orange"])
    ax.set_title(title)
    ax.set_xlabel("метод")
    ax.set_ylabel(metric_col)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return ax
