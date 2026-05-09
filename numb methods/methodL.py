import numpy as np
import matplotlib.pyplot as plt

# --- функція з варіанту 4 ---
def f(x):
    return np.sign(2 * x**3 - 4 * x) * np.abs(2 * x**3 - 4 * x)**(1/5)

# --- многочлен Лагранжа ---
def Lagrange(x, xs, ys):
    total = 0
    n = len(xs)
    for i in range(n):
        term = ys[i]
        for j in range(n):
            if i != j:
                term *= (x - xs[j]) / (xs[i] - xs[j])
        total += term
    return total

# --- параметри ---
a, b = 0, 8
n = 5  # кількість вузлів
x_nodes = np.linspace(a, b, n)
y_nodes = f(x_nodes)

# --- додаткові (середні) точки ---
x_mid = [(x_nodes[i] + x_nodes[i+1]) / 2 for i in range(len(x_nodes)-1)]

# --- об'єднані точки (вузли + середини) ---
x_all = np.unique(np.concatenate([x_nodes, np.array(x_mid)]))

y_f = [f(x) for x in x_all]
y_L = [Lagrange(x, x_nodes, y_nodes) for x in x_all]


# --- таблиця ---
print("x\ty\t\tL(x)")
print("-"*35)
for xi, fi, Li in zip(x_all, y_f, y_L):
    print(f"{xi:<.2f}\t{fi:<.6f}\t{Li:<.6f}")
# считаем кривые
x_plot = np.linspace(a - 1, b + 1, 400)
y_true = f(x_plot)
y_lagr = [Lagrange(x, x_nodes, y_nodes) for x in x_plot]



fig, ax = plt.subplots(figsize=(7, 7))      # квадратное полотно
ax.plot(x_plot, y_true, 'r', label='f(x)')
ax.plot(x_plot, y_lagr, 'b--', label='L(x) – Лагранж')
ax.scatter(x_nodes, y_nodes, color='black', zorder=5, label='вузли інтерполяції')
ax.scatter(x_all, [f(x) for x in x_all], color='red', s=40, label='f(x)')
ax.scatter(x_all, [Lagrange(x, x_nodes, y_nodes) for x in x_all],
            color='blue', marker='s', s=40, label='L(x)')

y_limit = (b - a + 4)/2  # замість L
ax.set_ylim(-y_limit, y_limit)
ax.set_xlim(a-2, b+2)
ax.set_aspect('equal', adjustable='box')

ax.grid(True)
ax.legend()
plt.show()