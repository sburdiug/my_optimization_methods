import numpy as np
import matplotlib.pyplot as plt
from method_progonki import p_forward, p_backward   # використовуємо твій метод прогонки

# ---------------------------------------------------------
#  ФУНКЦІЯ ТА ЇЇ ПОХІДНА
# ---------------------------------------------------------
def f(x):
    """Задана функція f(x) = (2x³ − 4x)^(1/5)"""
    v = 2*x**3 - 4*x
    return np.sign(v) * np.abs(v)**(1/5)

def fprime(x):
    """
    Аналітична похідна f'(x) з контролем ∞.
    Якщо x — скаляр, повертається звичайне число, а не масив.
    """
    x = np.atleast_1d(x)  # 🔹 робимо з x масив навіть якщо це скаляр
    v = 2*x**3 - 4*x
    dv = 6*x**2 - 4

    with np.errstate(divide="ignore", invalid="ignore"):
        der = (1/5) * dv * np.sign(v) * np.abs(v)**(-4/5)

    der[np.isnan(der)] = 0.0
    der[np.isinf(der)] = np.sign(der[np.isinf(der)]) * 1e6

    # якщо користувач передав скаляр, повертаємо скаляр
    return der[0] if der.size == 1 else der



# ---------------------------------------------------------
#  ПЕРШИЙ МЕТОД (через безпосереднє знаходження коефіцієнтів)
# ---------------------------------------------------------
def build_cubic_spline_first_method(xs, ys, fp0, fpN):
    """
    Побудова кубічного сплайна за першим методом:
    безпосередньо визначаються коефіцієнти поліномів S_i(x)
    на кожному інтервалі [x_i, x_{i+1}],
    використовуючи систему безперервності для S, S', S''.
    """
    n = len(xs)
    h = np.diff(xs)

    # невідомі: коефіцієнти c_i (другі похідні/3) у вузлах
    A_lower = np.zeros(n)
    A_main  = np.zeros(n)
    A_upper = np.zeros(n)
    rhs     = np.zeros(n)

    # крайові умови (зажаті)
    A_main[0]  = 2*h[0]
    A_upper[0] = h[0]
    rhs[0]     = 3*((ys[1]-ys[0])/h[0] - fp0)

    for i in range(1, n-1):
        A_lower[i] = h[i-1]
        A_main[i]  = 2*(h[i-1]+h[i])
        A_upper[i] = h[i]
        rhs[i]     = 3*((ys[i+1]-ys[i])/h[i] - (ys[i]-ys[i-1])/h[i-1])

    A_lower[-1] = h[-1]
    A_main[-1]  = 2*h[-1]
    rhs[-1]     = 3*(fpN - (ys[-1]-ys[-2])/h[-1])

    # розв'язуємо прогонкою
    alpha, gamma = p_forward(A_lower, A_main, A_upper, rhs)
    c = np.array(p_backward(alpha, gamma))  # тут c_i — це 1/3 S''(x_i)

    # далі обчислюємо решту коефіцієнтів (a_i,b_i,d_i)
    a = ys[:-1]
    b = np.zeros(n-1)
    d = np.zeros(n-1)

    for i in range(n-1):
        b[i] = (ys[i+1] - ys[i]) / h[i] - h[i]*(2*c[i] + c[i+1]) / 3
        d[i] = (c[i+1] - c[i]) / (3*h[i])

    return a, b, c[:-1], d, c


def spline_eval_first(xq, xs, a, b, c, d):
    """Обчислення значення сплайна (перший метод)"""
    xq = np.asarray(xq)
    if np.any((xq < xs[0]) | (xq > xs[-1])):
        raise ValueError("xq виходить за межі вузлів сплайна")
    idx = np.searchsorted(xs, xq, side='right') - 1
    idx = np.clip(idx, 0, len(a)-1)
    dx = xq - xs[idx]
    return a[idx] + b[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3


# ---------------------------------------------------------
#  ОСНОВНИЙ БЛОК ПРОГРАМИ
# ---------------------------------------------------------
if __name__ == "__main__":
    # Вихідні дані
    a0, b0 = -7.0, 2.0
    n_nodes = 11
    xs = np.linspace(a0, b0, n_nodes)
    ys = f(xs)

    # Похідні на кінцях
    fp0, fpN = fprime(xs[0]), fprime(xs[-1])

    # Побудова сплайна першим методом
    a_coef, b_coef, c_coef, d_coef, c_full = build_cubic_spline_first_method(xs, ys, fp0, fpN)

    # Обчислення для побудови графіка
    x_plot = np.linspace(a0, b0, 1200)
    y_true = f(x_plot)
    y_spl  = spline_eval_first(x_plot, xs, a_coef, b_coef, c_coef, d_coef)

    # Графік
    plt.figure(figsize=(9,6))
    plt.plot(x_plot, y_true, label='f(x) — вихідна функція', linewidth=2)
    plt.plot(x_plot, y_spl, '--', label='Sₙ(x) — кубічний сплайн (1-й метод)', alpha=0.9)
    plt.scatter(xs, ys, s=40, color='black', zorder=5, label=f'Вузли ({n_nodes} шт.)')
    plt.title("Кубічний сплайн (перший метод) для f(x)=√[5]{2x³−4x}")
    plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # Похибка
    err = np.max(np.abs(y_spl - y_true))
    print(f"Максимальна |похибка| на густій сітці: {err:.6g}")
