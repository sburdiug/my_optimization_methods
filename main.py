import numpy as np
import sympy as sp

from dfp_method import run_dfp_method
from fletcher_reeves_method import run_fletcher_reeves_method


if __name__ == "__main__":
    # Налаштування методу ДФП
    x1, x2 = sp.symbols("x1 x2")
    fx = 2 * x1**2 + x1 * x2 + x2**2
    x_start = (2, 2)
    a0 = np.eye(2)
    max_iter = 2

    run_dfp_method(
        fx=fx,
        x_start=x_start,
        a0=a0,
        max_iter=max_iter,
        show_plot=True,
    )
    print("="*50,"\n")
    # Налаштування методу Флетчера-Рівса
    x1, x2 = sp.symbols("x1 x2")
    fx = 2 * x1**2 + x1 * x2 + 2*x2**2 +8*x1
    x_start = (0, 0)
    max_iter = 2

    run_fletcher_reeves_method(
        fx=fx,
        x_start=x_start,
        max_iter=max_iter,
        show_plot=True,
    )
