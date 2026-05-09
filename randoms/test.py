from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANT4 = [
    2, 6, 8, 14, 15, 18, 22, 24, 28, 31, 35, 39, 42, 45, 46,
    51, 55, 60, 62, 64, 71, 73, 78, 81, 84, 88, 91, 96, 98, 99,
]


def sturges_k(n: int) -> int:
    return int(np.ceil(1 + np.log2(n)))


def load_data() -> np.ndarray:
    csv_path = Path(__file__).with_name("лаб1.csv")
    df = pd.read_csv(csv_path, header=None, sep=None, engine="python")
    df = df.replace({",": "."}, regex=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] > 10:
        df = df.iloc[:, -10:]

    if df.shape[1] != 10:
        raise ValueError(
            f"Expected 10 columns after cleanup, got {df.shape[1]}. "
            "Check delimiter and number format."
        )

    return df.to_numpy()


def plot_section(ax, values: np.ndarray, edges: np.ndarray, title: str) -> None:
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    heights = counts / widths

    ax.bar(
        edges[:-1],
        heights,
        width=widths,
        align="edge",
        edgecolor="black",
        linewidth=1.0,
    )
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{x:.2f}" for x in edges], rotation=45, ha="right")
    ax.set_ylabel("h = n_i / Δ")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def main() -> None:
    data = load_data()
    idx = [i - 1 for i in VARIANT4]
    selected = data[idx, :]

    n = selected.shape[0]
    k = sturges_k(n)
    print(f"k = {k}, n = {n}")
    output_dir = Path(__file__).parent

    for j in range(10):
        values = selected[:, j]
        local_min = float(np.min(values))
        local_max = float(np.max(values))

        if np.isclose(local_min, local_max):
            pad = 1e-6
            local_min -= pad
            local_max += pad

        edges_local = np.linspace(local_min, local_max, k + 1)
        delta = edges_local[1] - edges_local[0]
        print(f"section {j + 1}: min={local_min:.4f}, max={local_max:.4f}, delta={delta:.4f}")

        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True, facecolor="white")
        ax.set_facecolor("white")
        plot_section(ax, values, edges_local, f"Переріз {j + 1}")

        output_path = output_dir / f"section_{j + 1:02d}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
