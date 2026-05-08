import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 3 * (x[0] - 4) ** 2 + (x[1] - 2) ** 2
    #return 10 * (x[0] - 10) ** 2 + (x[1] - 10) ** 2

def nelder_mead_method(x1, x2, x3, params, max_iter=6):
    """Метод Нелдера-Міда з точним відтворенням логіки методички."""
    alpha = params["alpha"]
    beta_abs = abs(params["beta"])  # Модуль β; знак обирається автоматично (+/-)
    gamma = params["gamma"]
    M = params.get("M", 3)  # Критерій віку для редукції

    # Ініціалізація точок. Додаємо поле age для відстеження умови редукції
    points = [
        {'idx': 1, 'x': np.array(x1, dtype=float), 'age': 1},
        {'idx': 2, 'x': np.array(x2, dtype=float), 'age': 1},
        {'idx': 3, 'x': np.array(x3, dtype=float), 'age': 1},
    ]

    next_idx = 4
    history = []

    for k in range(1, max_iter + 1):
        # Обчислюємо значення функції
        for p in points:
            p['f'] = round(f(p['x']), 6)

        # Сортуємо вершини: l (найкраща/найменша), g (середня), h (найгірша/найбільша)
        points.sort(key=lambda p: p['f'])
        p_l, p_g, p_h = points[0], points[1], points[2]

        # Зберігаємо історію для графіка
        history.append([
            {'idx': p_l['idx'], 'x': p_l['x'].copy()},
            {'idx': p_g['idx'], 'x': p_g['x'].copy()},
            {'idx': p_h['idx'], 'x': p_h['x'].copy()},
        ])

        print(
            f"Ітерація {k}: "
            f"l=x^({p_l['idx']}) f={p_l['f']:.4f}, "
            f"g=x^({p_g['idx']}) f={p_g['f']:.4f}, "
            f"h=x^({p_h['idx']}) f={p_h['f']:.4f}"
        )

        # Перевірка умови редукції (M = 3)
        max_age = max(p['age'] for p in points)
        if max_age >= M:
            print(f"  Редукція: x^({p_l['idx']}) зберігається (M={M}).")

            # Редукція відносно найкращої точки x^(l)
            new_x_g = 0.5 * (p_g['x'] + p_l['x'])
            new_x_h = 0.5 * (p_h['x'] + p_l['x'])

            p_new_g = {'idx': next_idx, 'x': new_x_g, 'age': 1}
            print(f"  Нова точка x^({next_idx}) = ({new_x_g[0]:.2f}, {new_x_g[1]:.2f})")
            next_idx += 1

            p_new_h = {'idx': next_idx, 'x': new_x_h, 'age': 1}
            print(f"  Нова точка x^({next_idx}) = ({new_x_h[0]:.2f}, {new_x_h[1]:.2f})")
            next_idx += 1

            p_l['age'] = 1  # Скидаємо вік лідера після редукції
            points = [p_l, p_new_g, p_new_h]
            continue

        # Знаходимо центр тяжіння (x_c)
        x_c = 0.5 * (p_l['x'] + p_g['x'])
        print(f"  Центр x_c = ({x_c[0]:.2f}, {x_c[1]:.2f})")

        # Пробне симетричне відображення (alpha)
        x_new = x_c + alpha * (x_c - p_h['x'])
        f_new = round(f(x_new), 6)
        print(f"  Відображення x_new = ({x_new[0]:.2f}, {x_new[1]:.2f}), f={f_new:.4f}")

        if f_new < p_l['f']:
            # Розтягування (gamma)
            print(f"  Режим: розтягування (gamma={gamma}).")
            x_exp = p_h['x'] + (1 + gamma) * (x_c - p_h['x'])
            f_exp = round(f(x_exp), 6)
            print(f"  Нова точка x^({next_idx}) = ({x_exp[0]:.2f}, {x_exp[1]:.2f}), f={f_exp:.4f}")

            new_p = {'idx': next_idx, 'x': x_exp, 'age': 1}
            next_idx += 1
            points = [p_l, p_g, new_p]

        elif p_g['f'] <= f_new < p_h['f']:
            # Зовнішнє стискання: β = +|β|
            beta = beta_abs
            print(f"  Режим: стискання зовнішнє (beta={beta}).")
            x_con = p_h['x'] + (1 + beta) * (x_c - p_h['x'])
            f_con = round(f(x_con), 6)
            print(f"  Нова точка x^({next_idx}) = ({x_con[0]:.2f}, {x_con[1]:.2f}), f={f_con:.4f}")

            new_p = {'idx': next_idx, 'x': x_con, 'age': 1}
            next_idx += 1
            points = [p_l, p_g, new_p]

        elif f_new >= p_h['f']:
            # Внутрішнє стискання: β = -|β|
            beta = -beta_abs
            print(f"  Режим: стискання внутрішнє (beta={beta}).")
            x_con = p_h['x'] + (1 + beta) * (x_c - p_h['x'])
            f_con = round(f(x_con), 6)
            print(f"  Нова точка x^({next_idx}) = ({x_con[0]:.2f}, {x_con[1]:.2f}), f={f_con:.4f}")

            new_p = {'idx': next_idx, 'x': x_con, 'age': 1}
            next_idx += 1
            points = [p_l, p_g, new_p]

        else:
            print("  Режим: прийнято відображення.")
            new_p = {'idx': next_idx, 'x': x_new, 'age': 1}
            next_idx += 1
            points = [p_l, p_g, new_p]

        # Збільшуємо вік точок, які залишилися в багатограннику
        p_l['age'] += 1
        p_g['age'] += 1

    # Записуємо фінальний багатогранник
    for p in points: p['f'] = round(f(p['x']), 6)
    history.append([
        {'idx': points[0]['idx'], 'x': points[0]['x'].copy()},
        {'idx': points[1]['idx'], 'x': points[1]['x'].copy()},
        {'idx': points[2]['idx'], 'x': points[2]['x'].copy()},
    ])

    return points, history


def stopping_criteria(final_points):
    """Обчислення критерію закінчення за дисперсією значень функції."""
    print("\nКритерій зупинки:")

    pts = [p['x'] for p in final_points]
    fs = [p['f'] for p in final_points]

    # Центр тяжіння фінального багатогранника
    x_c = np.mean(pts, axis=0)
    f_c = f(x_c)

    print("  Фінальні вершини:")
    for p in final_points:
        print(f"  x^({p['idx']}) = ({p['x'][0]:.2f}, {p['x'][1]:.2f}), f={p['f']:.4f}")

    print(f"  Центр x_c = ({x_c[0]:.2f}, {x_c[1]:.2f}), f(x_c)={f_c:.4f}")

    variance = 0
    for value in fs:
        diff = value - f_c
        variance += diff ** 2

    variance /= 3
    crit = np.sqrt(variance)

    print(f"  Критерій = {crit:.4f}\n")


def plot_trajectory(history):
    """Графік деформації багатогранників."""
    fig, ax = plt.subplots(figsize=(8, 8))

    all_points = np.array([p['x'] for simplex in history for p in simplex])
    x_vals = all_points[:, 0]
    y_vals = all_points[:, 1]

    x_span = max(x_vals.max() - x_vals.min(), 1.0)
    y_span = max(y_vals.max() - y_vals.min(), 1.0)
    x_pad = 0.2 * x_span
    y_pad = 0.2 * y_span

    x_min = x_vals.min() - x_pad
    x_max = x_vals.max() + x_pad
    y_min = y_vals.min() - y_pad
    y_max = y_vals.max() + y_pad

    # Лінії рівня цільової функції
    x1 = np.linspace(x_min, x_max, 400)
    x2 = np.linspace(y_min, y_max, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f(np.array([X1, X2]))
    ax.contour(X1, X2, Z, levels=30, cmap='Blues', alpha=0.4)

    # Малюємо кожен багатогранник
    colors = plt.cm.viridis(np.linspace(0, 1, len(history)))
    for k, simplex in enumerate(history):
        simplex_coords = np.array([p['x'] for p in simplex])
        # Замикаємо трикутник (з'єднуємо останню точку з першою)
        simplex_closed = np.vstack((simplex_coords, simplex_coords[0]))

        lw = 2.5 if k == len(history) - 1 else 1.0  # Останній виділяємо товщиною
        alpha_val = 1.0 if k == len(history) - 1 else 0.6

        ax.plot(simplex_closed[:, 0], simplex_closed[:, 1], marker='o',
                color=colors[k], linewidth=lw, alpha=alpha_val, label=f'Ітерація {k + 1}')

        # Підписуємо вершини поточного багатогранника
        for p in simplex:
            ax.annotate(
                f"x^({p['idx']})",
                (p['x'][0], p['x'][1]),
                textcoords='offset points',
                xytext=(5, 5),
                fontsize=8,
                color=colors[k],
                alpha=alpha_val,
            )

    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Траєкторія стискання багатогранника (Нелдер-Мід)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    params = {"alpha": 1, "beta": 0.5, "gamma": 2, "M": 3}
    x1 = [7, 6]
    x2 = [9, 8]
    x3 = [7, 8]
    max_iter = 15

    final_points, history = nelder_mead_method(x1, x2, x3, params, max_iter)
    stopping_criteria(final_points)
    plot_trajectory(history)
