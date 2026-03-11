import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """Цільова функція: (x1 - 2)^2 + x2^2"""
    #return (x[0] - 2.0) ** 2 + x[1] ** 2
    #return (x[0] - 10.0) ** 2 + 10.0*x[1] ** 2
    return 2*(x[0]-4)**2 + x[1]**2

def exploratory_search(x, delta):
    x_new = x.copy()
    for i in range(len(x)):
        f_best = round(f(x_new), 6)

        x_new[i] += delta[i]
        if round(f(x_new), 6) < f_best:
            continue

        x_new[i] -= 2 * delta[i]
        if round(f(x_new), 6) < f_best:
            continue

        x_new[i] += delta[i]
    return x_new


def hooke_jeeves(x0, delta0, max_iter=6):
    x_bt = np.array(x0, dtype=float)
    delta = np.array(delta0, dtype=float)

    history = [x_bt.copy()]

    iter_num = 1  # Номер ітерації для логу (Ітерація 1, 2, 3...)
    bt_idx = 0  # Індекс Базисної Точки (x^0, x^1...)
    pbt_idx = 1  # Індекс Поточної Базисної Точки (x^1, x^2...)

    print(f"Початкова БТ: x^({bt_idx}) = ({x_bt[0]:.4f}, {x_bt[1]:.4f}), f = {f(x_bt):.4f}")

    x_pbt = exploratory_search(x_bt, delta)
    print(f"Ітерація {iter_num}, Знайдена ПБТ: x^({pbt_idx}) = ({x_pbt[0]:.4f}, {x_pbt[1]:.4f}), f = {f(x_pbt):.4f}")
    history.append(x_pbt.copy())

    while True:
        if iter_num >= max_iter:
            print(f" ліміт ітерацій: {max_iter}")
            break

        iter_num += 1
        probe_idx = pbt_idx + 1  # Наступна точка завжди на 1 більша за ПБТ

        x_p = 2 * x_pbt - x_bt
        x_probe = exploratory_search(x_p, delta)

        print(
            f"Ітерація {iter_num} отримана точка: x^({probe_idx}) = ({x_probe[0]:.4f}, {x_probe[1]:.4f}), f = {f(x_probe):.4f}")

        if round(f(x_probe), 6) < round(f(x_pbt), 6):
            print(
                f"   -> Успіх f(x^({probe_idx})) < f(x^({pbt_idx})). x^({pbt_idx}) стає БТ, x^({probe_idx}) стає ПБТ")
            x_bt = x_pbt.copy()
            x_pbt = x_probe.copy()
            history.append(x_pbt.copy())

            # Оновлюємо індекси точок
            bt_idx = pbt_idx
            pbt_idx = probe_idx
        else:
            print(f"   -> Невдача f(x^({probe_idx})) = {f(x_probe):.4f} > f(x^({pbt_idx})) = {f(x_pbt):.4f}.")
            print(f"   -> Відкидаємо ПБТ x^({pbt_idx}) та повертаємося до останньої БТ x^({bt_idx}) і зменшуємо крок")
            delta /= 2.0

            if len(history) > 1:
                history.pop()

            iter_num += 1
            pbt_idx = bt_idx + 1

            x_pbt = exploratory_search(x_bt, delta)
            print(
                f"Ітерація {iter_num}, пошук навколо x^({bt_idx}) дає нову ПБТ: x^({pbt_idx}) = ({x_pbt[0]:.4f}, {x_pbt[1]:.4f}), f = {f(x_pbt):.4f}")
            history.append(x_pbt.copy())


    # Повертаємо знайдені точки та їхні індекси для правильного розрахунку критеріїв
    return x_bt, x_pbt, history, delta, bt_idx, pbt_idx


def stopping_criteria(x_bt, x_pbt, delta, bt_idx, pbt_idx):

    print(
        f"Використовуємо точки:\nБТ x^({bt_idx}) = ({x_bt[0]:.4f}, {x_bt[1]:.4f})\nПБТ x^({pbt_idx}) = ({x_pbt[0]:.4f}, {x_pbt[1]:.4f})\nКрок Δx = ({delta[0]:.4f}, {delta[1]:.4f})\n")

    # Спосіб 1
    delta_norm = np.linalg.norm(delta)
    print(f"Спосіб 1: ||Δx||")
    print(f"||Δx|| = sqrt({delta[0]}^2 + {delta[1]}^2) = {delta_norm:.4f}")
    print()

    # Спосіб 2
    norm_x_prev = np.linalg.norm(x_bt)
    norm_diff_x = np.linalg.norm(x_pbt - x_bt)
    rel_x = norm_diff_x / max(norm_x_prev, 1e-12)

    f_prev = f(x_bt)
    f_curr = f(x_pbt)
    rel_f = abs(f_curr - f_prev) / max(abs(f_prev), 1e-12)

    print(f"Спосіб 2:")
    print(f"1) ||x^({pbt_idx}) - x^({bt_idx})|| / ||x^({bt_idx})||")
    print(f"   {norm_diff_x:.4f} / {norm_x_prev:.4f} = {rel_x:.4f}")

    print(f"2) |f(x^({pbt_idx})) - f(x^({bt_idx}))| / |f(x^({bt_idx}))|")
    print(f"   |{f_curr:.4f} - {f_prev:.4f}| / |{f_prev:.4f}| = {rel_f:.4f}")



def plot_trajectory(history):
    history = np.array(history)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(history[:, 0], history[:, 1], "o-", color="tab:blue", linewidth=1.8, markersize=5)

    for i, point in enumerate(history):
        ax.annotate(f"x^{i}", (point[0], point[1]), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.scatter(history[0, 0], history[0, 1], color="tab:green", s=60, marker="s", label="Старт")
    ax.scatter(history[-1, 0], history[-1, 1], color="tab:red", s=60, marker="o", label="Фініш")

    x_min, x_max = np.min(history[:, 0]) - 0.5, np.max(history[:, 0]) + 0.5
    y_min, y_max = np.min(history[:, 1]) - 0.5, np.max(history[:, 1]) + 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    step = 1.0
    ax.set_xticks(np.arange(np.floor(x_min), np.ceil(x_max) + step, step))
    ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max) + step, step))

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Траєкторія ітерацій методу Хука-Дживса")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    x0 = [6.0, 6.0]
    #x0= [10.0, 10.0]
    delta0 = [0.6, 0.8]

    x_bt, x_pbt, hist, delta_final, bt_idx, pbt_idx = hooke_jeeves(x0, delta0, max_iter=12)

    stopping_criteria(x_bt, x_pbt, delta_final, bt_idx, pbt_idx)

    plot_trajectory(hist)
