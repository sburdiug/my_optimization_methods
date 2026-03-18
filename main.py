import sympy as sp

from steepest_descent_constant_step import run_steepest_descent_constant_step
from steepest_descent_optimal_step import run_steepest_descent_optimal_step
from partan_steepest_descent_method import run_partan_mns
from newton_multivariate_method import run_newton_multivariate_method


if __name__ == "__main__":
    # Налаштування методів у main.py
    x1, x2 = sp.symbols("x1 x2")
    max_iter = 3

    fx = 3 * (x1 - 4)** 2 + x1 * x2 + 7 * x2 ** 2



    #fx_const = 3 * x1**2 + x1 * x2 + 2 * x2**2
    x_start_const = (9.8, 9.8)
    step = 1.0
    run_steepest_descent_constant_step(
        fx=fx,
        x_start=x_start_const,
        step=step,
        max_iter=max_iter,
        show_plot=True,
    )

    print("\n" + "=" * 60 + "\n")

    #fx_opt = 4 * x1**2 + x1 * x2 + x2**2
    x_start_opt = (9.8, 9.8)
    run_steepest_descent_optimal_step(
        fx=fx,
        x_start=x_start_opt,
        max_iter=max_iter,
        show_plot=True,
    )

    print("\n" + "=" * 60 + "\n")

    #fx_partan = 4 * x1**2 + x1 * x2 + x2**2
    run_partan_mns(
        fx=fx,
        x_start=(9.8, 9.8),
        max_iter=max_iter,
        eps=1e-4,
        show_plot=True,
    )

    print("\n" + "=" * 60 + "\n")

    #fx_newton = 4 * x1**2 + x1 * x2 + x2**2
    run_newton_multivariate_method(
        fx=fx,
        x_start=(9.8, 9.8),
        max_iter=max_iter,
        eps=1e-4,
        show_plot=True,
    )
