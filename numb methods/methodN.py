import numpy as np
from math import factorial

# --- функція варіанту 4: реальний 5-й корінь ---
def f(x):
    v = 2*x**3 - 4*x
    return np.sign(v) * np.abs(v)**(1/5)

# --- різниці вперед: повертає всю трикутну таблицю (як списки рядків) ---
def forward_diffs_full(y):
    y = np.asarray(y, dtype=float)
    table = [y]
    while table[-1].size > 1:
        table.append(np.diff(table[-1]))
    return table  # table[k] = Δ^k y_i, i=0..m-k

# --- перший стовпчик різниць: [Δ^0 y0, Δ^1 y0, ..., Δ^m y0] ---
def first_column_forward_diffs(y):
    table = forward_diffs_full(y)
    return np.array([row[0] for row in table], dtype=float)

def falling_fact(t, k):
    """t(t-1)...(t-k+1), FF(t,0)=1"""
    out = 1.0
    for j in range(k):
        out *= (t - j)
    return out

# --- оцінка через формулу Ньютона вперед: повертає терми N0..Nm і часткові суми L0..Lm ---
def newton_forward_terms(x, x0, h, diffs0):
    t = (x - x0)/h
    terms = []
    for k, delta in enumerate(diffs0):
        terms.append(falling_fact(t, k)/factorial(k) * delta)
    partial = np.cumsum(terms)
    return t, terms, partial  # t, [N0..Nm], [L0..Lm]

# ================= ПАРАМЕТРИ (можеш міняти) =================
a, b = -2.0, 2.0    # або 0.0, 8.0 якщо хочеш приклад з h=2
n = 5               # 5 вузлів
xs = np.linspace(a, b, n)
ys = f(xs)
h = xs[1] - xs[0]

# 1) Таблиця прямих різниць (як у твоєму скріні зверху)
full = forward_diffs_full(ys)
print("=== Таблиця прямих різниць (вперед) ===")
max_width = len(full[0])
headers = [f"Δ^{k} y" if k>0 else "y" for k in range(len(full))]
print("\t".join(headers))
for i in range(max_width):
    row = []
    for k in range(len(full)):
        if i < len(full[k]):
            row.append(f"{full[k][i]: .10f}")
        else:
            row.append("")
    print("\t".join(row))
print()

# 2) Таблиця N0..N4 і L0..L4 (як «Лагранж-таблиця», але для Ньютона)
diffs0 = first_column_forward_diffs(ys)      # Δ^k y0
x_mid = (xs[:-1] + xs[1:]) / 2               # середини
x_all = np.sort(np.concatenate([xs, x_mid])) # вузли + середини

print("=== Оцінка у точках (вузли + середини) — Ньютон вперед ===")
hdr = ["x", "t=(x-x0)/h", "N0", "N1", "N2", "N3", "N4", "L0", "L1", "L2", "L3", "L4", "N(x)", "f(x)"]
print("\t".join(hdr))
for x in x_all:
    t, terms, partial = newton_forward_terms(x, xs[0], h, diffs0)
    # обрізаємо/доповнюємо до 5 термів (бо 5 вузлів -> степінь ≤ 4)
    terms  = (terms + [0.0]*5)[:5]
    partial = (partial.tolist() + [partial[-1]]*5)[:5]
    Nx = partial[-1]
    fx = f(x)
    row = [f"{x: .6f}", f"{t: .6f}"] + [f"{v: .10f}" for v in terms] + [f"{v: .10f}" for v in partial] + [f"{Nx: .10f}", f"{fx: .10f}"]
    print("\t".join(row))
