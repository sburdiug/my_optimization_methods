import numpy as np


def circle_constraint(x):
    """
    Обмеження для умовної оптимізації:
    x1^2 + x2^2 <= 1.

    У коді використовується форма g(x) <= 0:
    g(x) = x1^2 + x2^2 - 1.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x1, x2 = x
    return float(x1**2 + x2**2 - 1.0)


def constraint_violation(g_value):
    """
    Порушення для обмеження g(x) <= 0.
    """
    return max(0.0, float(g_value))


def squared_penalty(x, constraints):
    """
    Штраф P(x) = sum(max(0, g_i(x))^2).
    """
    return float(sum(constraint_violation(g(x)) ** 2 for g in constraints))


def make_external_penalty_function(f, constraints, r):
    """
    Формує штрафну функцію зовнішньої точки:
    F(x, r) = f(x) + r * P(x).
    """
    if r <= 0:
        raise ValueError("r має бути додатним.")

    def penalty_function(x):
        return float(f(x) + r * squared_penalty(x, constraints))

    return penalty_function


def total_violation(x, constraints):
    """
    Сума звичайних порушень обмежень без квадрата.
    """
    return float(sum(constraint_violation(g(x)) for g in constraints))
