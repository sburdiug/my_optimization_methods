import numpy as np
import matplotlib.pyplot as plt

from rk4 import rk4


def kolmogorov_system(t, p):
    p1, p2, p3, p4, p5 = p
    dp1 = 1.0 * p2 + 1.0 * p3 + 2.0 * p4 - 3.0 * p1
    dp2 = 3.0 * p1 - (1.0 + 4.0 + 2.0) * p2
    dp3 = 4.0 * p2 + 1.0 * p4 - (1.0 + 3.0 + 2.0) * p3
    dp4 = 3.0 * p3 + 2.0 * p5 - (1.0 + 2.0) * p4
    dp5 = 2.0 * p2 + 2.0 * p3 - 2.0 * p5
    return np.array([dp1, dp2, dp3, dp4, dp5], dtype=float)


def estimate_t_critical(ts, probs, eps=1e-4, window=30):
    if len(ts) <= window:
        return ts[-1]

    deltas = np.max(np.abs(np.diff(probs, axis=0)), axis=1)
    calm = deltas < eps

    streak = 0
    for i, ok in enumerate(calm):
        if ok:
            streak += 1
            if streak >= window:
                return ts[i - window + 2]
        else:
            streak = 0
    return ts[-1]


def plot_probabilities(ts, probs, t_critical, graph_type="plot"):
    if graph_type not in {"plot", "step"}:
        raise ValueError("graph_type must be 'plot' or 'step'")

    labels = ["p1(t)", "p2(t)", "p3(t)", "p4(t)", "p5(t)"]
    plt.figure(figsize=(10, 6))

    for i in range(probs.shape[1]):
        if graph_type == "step":
            plt.step(ts, probs[:, i], where="post", label=labels[i], linewidth=2)
        else:
            plt.plot(ts, probs[:, i], label=labels[i], linewidth=2)

    plt.axvline(t_critical, color="black", linestyle="--", linewidth=1.4, label=f"tкр ≈ {t_critical:.3f}")
    plt.xlabel("t")
    plt.ylabel("Ймовірність стану")
    plt.title("Ймовірності станів системи у часі (метод Рунге-Кутти 4-го порядку)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    t0 = 0.0
    tn = 10.0
    h = 0.01
    p0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    graph_type = "plot"  # "plot" або "step"

    ts, probs = rk4(kolmogorov_system, t0, p0, tn, h)
    t_critical = estimate_t_critical(ts, probs)

    print(f"Сума ймовірностей на старті: {np.sum(probs[0]):.6f}")
    print(f"Сума ймовірностей в кінці:  {np.sum(probs[-1]):.6f}")
    print(f"Орієнтовний tкр: {t_critical:.3f}")

    plot_probabilities(ts, probs, t_critical, graph_type=graph_type)


if __name__ == "__main__":
    main()
