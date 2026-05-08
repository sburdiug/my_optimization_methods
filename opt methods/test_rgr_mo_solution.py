import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import rgr_mo


def analytic_minimum(n):
    x1, x2 = sp.symbols("x1 x2", real=True)
    f = 3 * (x1 - n) ** 2 + x1 * x2 + 7 * x2**2

    equations = [sp.diff(f, x1), sp.diff(f, x2)]
    solutions = sp.solve(equations, (x1, x2), dict=True)
    if not solutions:
        raise ValueError("SymPy could not find a stationary point.")

    x_star = solutions[0]
    return float(x_star[x1]), float(x_star[x2])


def run_rgr_flow():
    rgr_mo.SHOW_SVEN_PLOT = False
    rgr_mo.SHOW_TRAJECTORY_PLOT = False

    x0 = np.array([1.2 * rgr_mo.N + 5, 1.2 * rgr_mo.N + 5], dtype=float)
    s1 = np.array([1.0, 0.0], dtype=float)
    s2 = np.array([0.0, 1.0], dtype=float)

    delta0 = rgr_mo.compute_delta(x0, s2)
    phi1 = rgr_mo.make_phi(x0, s2)
    interval0 = rgr_mo.get_sven_interval(phi1, delta0)
    lambda0, _ = rgr_mo.line_search("golden", phi1, interval0, rgr_mo.EPS)
    x1 = rgr_mo.next_point(x0, s2, lambda0)

    delta1 = rgr_mo.compute_delta(x1, s1)
    phi2 = rgr_mo.make_phi(x1, s1)
    interval1 = rgr_mo.get_sven_interval(phi2, delta1)
    lambda1, _ = rgr_mo.line_search("golden", phi2, interval1, rgr_mo.EPS)
    x2 = rgr_mo.next_point(x1, s1, lambda1)

    delta2 = rgr_mo.compute_delta(x2, s2)
    phi3 = rgr_mo.make_phi(x2, s2)
    interval2 = rgr_mo.get_sven_interval(phi3, delta2)
    lambda2, _ = rgr_mo.line_search("powell", phi3, interval2, rgr_mo.EPS)
    x3 = rgr_mo.next_point(x2, s2, lambda2)

    s4 = x3 - x1
    delta3 = rgr_mo.compute_delta(x3, s4)
    phi4 = rgr_mo.make_phi(x3, s4)
    interval3 = rgr_mo.get_sven_interval(phi4, delta3)
    lambda3, _ = rgr_mo.line_search("powell", phi4, interval3, rgr_mo.EPS)
    x4 = rgr_mo.next_point(x3, s4, lambda3)

    points = [x0, x1, x2, x3, x4]
    return x4, points


def plot_test_trajectory(points, x_star):
    px = [float(point[0]) for point in points]
    py = [float(point[1]) for point in points]

    xs = px + [x_star[0]]
    ys = py + [x_star[1]]
    margin = 1.0
    grid_x = np.linspace(min(xs) - margin, max(xs) + margin, 300)
    grid_y = np.linspace(min(ys) - margin, max(ys) + margin, 300)
    xx, yy = np.meshgrid(grid_x, grid_y)
    zz = rgr_mo.objective(np.stack([xx, yy], axis=0))

    plt.figure(figsize=(9, 6))
    plt.contour(xx, yy, zz, levels=30, cmap="viridis")
    plt.plot(px, py, "o-r", linewidth=2, markersize=6, label="trajectory x^0..x^4")
    for idx, point in enumerate(points):
        plt.annotate(f"x^{idx}", (point[0], point[1]), xytext=(5, 5), textcoords="offset points")

    plt.scatter([x_star[0]], [x_star[1]], c="blue", marker="*", s=180, label="analytic x*")
    plt.title("Test Trajectory vs Analytic Minimum")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    x4, points = run_rgr_flow()
    x_star = analytic_minimum(rgr_mo.N)

    err = float(np.linalg.norm(x4 - np.array(x_star, dtype=float)))
    f_x4 = float(rgr_mo.objective(x4))
    f_x_star = float(rgr_mo.objective(x_star))

    print(f"x* (analytic) = ({x_star[0]:.6f}, {x_star[1]:.6f})")
    print(f"x4 (rgr_mo)   = ({x4[0]:.6f}, {x4[1]:.6f})")
    print(f"||x4 - x*||   = {err:.6e}")
    print(f"f(x*)         = {f_x_star:.6f}")
    print(f"f(x4)         = {f_x4:.6f}")

    tolerance = 5e-3
    if err > tolerance:
        raise AssertionError(
            f"Solution check failed: error {err:.6e} is greater than tolerance {tolerance:.1e}"
        )

    plot_test_trajectory(points, x_star)
    print("PASS: rgr_mo solution matches analytic minimum within tolerance.")
