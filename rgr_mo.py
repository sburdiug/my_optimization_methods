
import matplotlib.pyplot as plt
import numpy as np

from golden_section_method import golden_section_method
from powell_method import powell_method
from sven import sven_method, visualize_sven


N = 4
EPS = 0.01
LAMBDA0 = 0.0
DELTA_BASE = 0.1
SHOW_SVEN_PLOT = False
SHOW_TRAJECTORY_PLOT = True


def objective(x):
    x_arr = np.asarray(x, dtype=float)
    x1, x2 = x_arr
    return 3 * (x1 - N) ** 2 + x1 * x2 + 7 * x2**2



def make_phi(xk, direction):
    xk = np.asarray(xk, dtype=float)
    direction = np.asarray(direction, dtype=float)

    def phi(lmbd):
        lam = np.asarray(lmbd, dtype=float)
        if lam.ndim == 0:
            return objective(xk + lam * direction)
        return objective(xk + lam[:, None] * direction)

    return phi


def compute_delta(xk, direction):
    return DELTA_BASE * (np.linalg.norm(xk) / np.linalg.norm(direction))


def get_sven_interval(phi, delta_lambda):
    interval, sven_points = sven_method(phi, LAMBDA0, delta_lambda)
    if SHOW_SVEN_PLOT:
        visualize_sven(phi, interval, sven_points, LAMBDA0, delta_lambda)
    return interval


def line_search(method, phi, interval, eps):
    if method == "golden":
        result_interval, _ = golden_section_method(phi, interval[0], interval[1], eps)
    elif method == "powell":
        result_interval = powell_method(phi, interval[0], interval[1], eps)
    else:
        raise ValueError("method must be 'golden' or 'powell'")

    return np.mean(result_interval), result_interval


def next_point(xk, direction, lmbd):
    return xk + lmbd * direction


def print_iteration(iteration_idx, method_name, xk, direction, delta_lambda, src_interval, result_interval, lmbd, x_next):
    print(f"\nІтерація {iteration_idx} ({method_name}):")
    print(f"x^{iteration_idx - 1} = ({xk[0]:.3f}, {xk[1]:.3f})")
    print(f"s = ({direction[0]:.3f}, {direction[1]:.3f})")
    if iteration_idx == 4:
        print(f"x3-x1 = ({direction[0]:.3f}, {direction[1]:.3f})")
    print(f"Δλ = {delta_lambda:.3f}")
    print(f"Інтервал невизначеності = [{src_interval[0]:.3f}, {src_interval[1]:.3f}]")
    if method_name.lower() == "dsk-powell":
        x1 = src_interval[0]
        x3 = src_interval[1]
        x2 = 0.5 * (x1 + x3)
        print(f"x1 = {x1:.3f}, x2 = {x2:.3f}, x3 = {x3:.3f}, x* = {lmbd:.3f}")
    else:
        print(f"Вихідний інтервал = [{result_interval[0]:.3f}, {result_interval[1]:.3f}]")
    print(f"λ_{iteration_idx - 1} = {lmbd:.3f}")
    print(f"x^{iteration_idx} = ({x_next[0]:.3f}, {x_next[1]:.3f})")


def plot_trajectory(points):
    if not SHOW_TRAJECTORY_PLOT:
        return

    pts = np.array(points)
    px, py = pts[:, 0], pts[:, 1]

    margin = 2.0
    grid_x = np.linspace(min(px) - margin, max(px) + margin, 250)
    grid_y = np.linspace(min(py) - margin, max(py) + margin, 250)
    xx, yy = np.meshgrid(grid_x, grid_y)
    zz = objective(np.stack([xx, yy], axis=0))

    plt.figure(figsize=(8, 6))
    plt.contour(xx, yy, zz, levels=25, cmap="viridis")
    plt.plot(px, py, "o-r", linewidth=2, markersize=6)
    for idx, (x_val, y_val) in enumerate(points):
        plt.annotate(f"x^{idx}", (x_val, y_val), xytext=(5, 5), textcoords="offset points")
    plt.title("Траєкторія пошуку")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    x0 = np.array([1.2 * N + 5, 1.2 * N + 5], dtype=float)
    s1 = np.array([1.0, 0.0], dtype=float)
    s2 = np.array([0.0, 1.0], dtype=float)

    print("=" * 70)
    print(f"n = {N}")
    print("f(x) = 3*(x1 - n)^2 + x1*x2 + 7*x2^2")
    print(f"x^(0) = ({x0[0]:.3f}, {x0[1]:.3f})")
    print(f"eps = {EPS:.3f}")
    print("=" * 70)

    # Sven + Golden/Powell on each iteration.
    delta0 = compute_delta(x0, s2)
    phi1 = make_phi(x0, s2)

    interval0 = get_sven_interval(phi1, delta0)

    # Iteration 1: Golden, direction S^(2)
    lambda0, interval1 = line_search("golden", phi1, interval0, EPS)
    x1 = next_point(x0, s2, lambda0)
    print_iteration(1, "golden", x0, s2, delta0, interval0, interval1, lambda0, x1)

    # Iteration 2: Golden, direction S^(1)
    delta1 = compute_delta(x1, s1)
    phi2 = make_phi(x1, s1)
    print("=" * 70)
    interval_sven_2 = get_sven_interval(phi2, delta1)
    lambda1, interval2 = line_search("golden", phi2, interval_sven_2, EPS)
    x2 = next_point(x1, s1, lambda1)
    print_iteration(2, "golden", x1, s1, delta1, interval_sven_2, interval2, lambda1, x2)

    # Iteration 3: DSK-Powell, direction S^(2)
    delta2 = compute_delta(x2, s2)
    phi3 = make_phi(x2, s2)
    print("=" * 70)
    interval_sven_3 = get_sven_interval(phi3, delta2)
    lambda2, interval3 = line_search("powell", phi3, interval_sven_3, EPS)
    x3 = next_point(x2, s2, lambda2)
    print_iteration(3, "dsk-powell", x2, s2, delta2, interval_sven_3, interval3, lambda2, x3)

    # Iteration 4: DSK-Powell, direction (x^3 - x^1)
    s4 = x3 - x1
    if np.linalg.norm(s4) <= 1e-12:
        raise ValueError("Direction (x^3 - x^1) is near zero; cannot continue iteration 4.")
    delta3 = compute_delta(x3, s4)
    phi4 = make_phi(x3, s4)
    print("=" * 70)
    interval_sven_4 = get_sven_interval(phi4, delta3)
    lambda3, interval4 = line_search("powell", phi4, interval_sven_4, EPS)
    x4 = next_point(x3, s4, lambda3)
    print_iteration(4, "dsk-powell", x3, s4, delta3, interval_sven_4, interval4, lambda3, x4)

    plot_trajectory([x0, x1, x2, x3, x4])

    to_float3 = lambda value: float(np.round(value, 3))
    print("\nПідсумок:")
    print(
        "Інтервали Свена: "
        f"I1=[{interval0[0]:.3f}, {interval0[1]:.3f}], "
        f"I2=[{interval_sven_2[0]:.3f}, {interval_sven_2[1]:.3f}], "
        f"I3=[{interval_sven_3[0]:.3f}, {interval_sven_3[1]:.3f}], "
        f"I4=[{interval_sven_4[0]:.3f}, {interval_sven_4[1]:.3f}]"
    )
    print(f"x^1 = ({x1[0]:.3f}, {x1[1]:.3f})")
    print(f"x^2 = ({x2[0]:.3f}, {x2[1]:.3f})")
    print(f"x^3 = ({x3[0]:.3f}, {x3[1]:.3f})")
    print(f"x^4 = ({x4[0]:.3f}, {x4[1]:.3f})")
    print(
        f"lambda0 = {lambda0:.3f}, lambda1 = {lambda1:.3f}, "
        f"lambda2 = {lambda2:.3f}, lambda3 = {lambda3:.3f}"
    )
