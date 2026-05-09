import numpy as np
import sympy as sp

from dfp_method import run_dfp_method
from fletcher_reeves_method import run_fletcher_reeves_method


if __name__ == "__main__":

    x1, x2 = sp.symbols("x1 x2")
    fx = 3 * (x1-4)**2 + x1 * x2 + 7*x2**2
    x_start = (9.8, 9.8)
    a0 = np.eye(2)

    # Налаштування методу Флетчера-Рівса
    run_fletcher_reeves_method(
        fx=fx,
        x_start=x_start,
        max_iter=2,
        show_plot=True,
    )

    print("="*50,"\n")
    # Налаштування методу ДФП
    run_dfp_method(
        fx=fx,
        x_start=x_start,
        a0=a0,
        max_iter=2,
        show_plot=True,
    )
