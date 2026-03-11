from task1_kr1 import run_task1
from task2_kr1 import run_task2


def f(x):
    """Функція f(x) = (100 - x)^2"""
    #return (100 - x) ** 2
    #return x**2 - 20*x
    #return x*(2*x-3)
    return x**2 - 5 * x


# Основна програма
if __name__ == "__main__":

    x0 = 3.5
    delta = 0.1
    eps = 0.2
    eps_golden = 0.2
    eps_powell = 0.01

    run_task1(f, x0, delta, eps, eps_golden, eps_powell)
    print()
    run_task2()

