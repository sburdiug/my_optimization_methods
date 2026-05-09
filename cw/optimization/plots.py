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
