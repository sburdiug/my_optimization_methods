import numpy as np
import matplotlib.pyplot as plt
from method_progonki import p_forward, p_backward


def f(x):
    """f(x) = (2x³ − 4x)^(1/5)"""
    v = 2*x**3 - 4*x
    return np.sign(v) * np.abs(v)**(1/5)

def fprime(x):
    """f'(x)"""
    v = 2*x**3 - 4*x
    dv = 6*x**2 - 4
    # избегаем деления на ноль
    return (1/5) * np.where(v != 0, dv * np.abs(v)**(-4/5), 0.0)

# --- параметры задачи ---
a, b = -5.0, 2.0
n_nodes = 11
xs = np.linspace(a, b, n_nodes)
ys = f(xs)

# зажатые края (clamped)
fp0 = fprime(xs[0])
fpN = fprime(xs[-1])

# --- сборка трёхдиагональной системы A*m = rhs ---
h = np.diff(xs)
N = len(xs)

A_lower = np.zeros(N)
A_main  = np.zeros(N)
A_upper = np.zeros(N)
rhs     = np.zeros(N)

# Левая граница (clamped)
A_main[0]  = 2*h[0]
A_upper[0] = h[0]
rhs[0]     = 6*((ys[1]-ys[0])/h[0] - fp0)

# Внутренние узлы
for i in range(1, N-1):
    A_lower[i] = h[i-1]
    A_main[i]  = 2*(h[i-1] + h[i])
    A_upper[i] = h[i]
    rhs[i]     = 6*((ys[i+1]-ys[i])/h[i] - (ys[i]-ys[i-1])/h[i-1])

# Правая граница (clamped)
A_lower[-1] = h[-1]
A_main[-1]  = 2*h[-1]
rhs[-1]     = 6*(fpN - (ys[-1]-ys[-2])/h[-1])

# --- решение методом прогонки из твоего файла ---
# (всё по твоим обозначениям: a, b, c, d)
alph, gamm = p_forward(A_lower, A_main, A_upper, rhs)
m = np.array(p_backward(alph, gamm))

# --- коэффициенты кубического сплайна ---
a_coef = ys[:-1]
b_coef = (ys[1:] - ys[:-1]) / h - h*(2*m[:-1] + m[1:]) / 6
c_coef = m[:-1] / 2
d_coef = (m[1:] - m[:-1]) / (6*h)

def spline_eval(xq):
    """Оценка кубического сплайна"""
    xq = np.asarray(xq)
    idx = np.searchsorted(xs, xq, side='right') - 1
    idx = np.clip(idx, 0, len(a_coef)-1)
    dx = xq - xs[idx]
    return a_coef[idx] + b_coef[idx]*dx + c_coef[idx]*dx**2 + d_coef[idx]*dx**3


# --- график ---
x_plot = np.linspace(a, b, 1200)
y_true = f(x_plot)
y_spl  = spline_eval(x_plot)

plt.figure(figsize=(9,6))
plt.plot(x_plot, y_true, label='f(x) = функція', linewidth=2)
plt.plot(x_plot, y_spl, '--', label='Sn(x) = кубічний сплайн', alpha=0.9)
plt.scatter(xs, ys, s=40, color='black', label='вузли')
plt.title("Кубический сплайн f(x)=√5{2x^3−4x}")
plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()

# --- проверка ---
err = np.max(np.abs(y_spl - y_true))
print(f"Максимальная |погрешность| на сетке: {err:.6g}")
