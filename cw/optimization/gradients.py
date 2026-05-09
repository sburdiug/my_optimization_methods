import numpy as np


def numerical_gradient(
    f,
    x: np.ndarray | list[float] | tuple[float, ...],
    h: float = 1e-4,
    scheme: str = "central",
) -> np.ndarray:
    if h <= 0:
        raise ValueError("h має бути додатним.")
    if scheme not in {"forward", "backward", "central"}:
        raise ValueError("scheme має бути одним із: forward, backward, central.")

    x_arr = np.asarray(x, dtype=float).reshape(-1)
    grad = np.zeros_like(x_arr, dtype=float)
    f_x = float(f(x_arr)) if scheme in {"forward", "backward"} else None

    for i in range(x_arr.size):
        step = np.zeros_like(x_arr)
        step[i] = h

        if scheme == "forward":
            grad[i] = (float(f(x_arr + step)) - f_x) / h
        elif scheme == "backward":
            grad[i] = (f_x - float(f(x_arr - step))) / h
        else:
            grad[i] = (float(f(x_arr + step)) - float(f(x_arr - step))) / (2.0 * h)

    return grad
