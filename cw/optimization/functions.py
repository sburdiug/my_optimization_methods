import numpy as np


FUNCTION_FORMULA = "f2(x) = (10*(x1 - x2)^2 + (x1 - 1)^2)^4"


def power_function(x: np.ndarray | list[float] | tuple[float, ...]) -> float:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    if x_arr.size != 2:
        raise ValueError("power_function очікує двовимірний вектор [x1, x2].")
    x1, x2 = x_arr
    return float((10.0 * (x1 - x2) ** 2 + (x1 - 1.0) ** 2) ** 4)


X_START = np.array([-1.2, 0.0], dtype=float)
X_MIN = np.array([1.0, 1.0], dtype=float)
F_MIN = 0.0


class FunctionCounter:
    def __init__(self, f):
        self.f = f
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.f(x)

    def reset(self):
        self.calls = 0
