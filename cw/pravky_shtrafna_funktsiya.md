# Правки для пункту про метод штрафних функцій

## 1. Що треба додати

Після дослідження МНС і ПАРТАН-МНС для безумовної оптимізації треба додати окремий блок для умовної оптимізації:

> Використати метод штрафних функцій, тобто метод зовнішньої точки, для випадку, коли локальний мінімум знаходиться поза випуклою допустимою областю.

Для цього залишаємо ту саму цільову функцію:

```text
f(x) = (10 * (x1 - x2)^2 + (x1 - 1)^2)^4
```

Безумовний мінімум цієї функції:

```text
x* = (1; 1)
f(x*) = 0
```

Допустиму область беремо таку:

```text
x1^2 + x2^2 <= 1
```

Це випукла область, бо це круг. Точка `(1; 1)` не належить цій області, бо:

```text
1^2 + 1^2 = 2 > 1
```

Отже, умова з постановки задачі виконується: локальний мінімум знаходиться поза випуклою допустимою областю.

---

## 2. Додати файл `optimization/penalty.py`

Створи новий файл:

```text
optimization/penalty.py
```

Код:

```python
import numpy as np


def circle_constraint(x):
    """
    Обмеження для умовної оптимізації:
    x1^2 + x2^2 <= 1

    У коді використовуємо форму:
    g(x) <= 0

    g(x) = x1^2 + x2^2 - 1
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x1, x2 = x
    return float(x1**2 + x2**2 - 1.0)


def constraint_violation(g_value):
    """
    Для обмеження g(x) <= 0:
    - якщо g(x) <= 0, порушення немає;
    - якщо g(x) > 0, точка поза допустимою областю.
    """
    return max(0.0, float(g_value))


def squared_penalty(x, constraints):
    """
    Штраф:
    P(x) = sum(max(0, g_i(x))^2)
    """
    return float(sum(constraint_violation(g(x)) ** 2 for g in constraints))


def make_external_penalty_function(f, constraints, r):
    """
    Формує штрафну функцію зовнішньої точки:

    F(x, r) = f(x) + r * P(x)
    """
    if r <= 0:
        raise ValueError("r має бути додатним.")

    def penalty_function(x):
        return float(f(x) + r * squared_penalty(x, constraints))

    return penalty_function


def total_violation(x, constraints):
    """
    Сума звичайних порушень обмежень без квадрата.
    Потрібно для таблиці результатів.
    """
    return float(sum(constraint_violation(g(x)) for g in constraints))
```

---

## 3. Оновити `optimization/__init__.py`

Додай імпорти:

```python
from .penalty import (
    circle_constraint,
    make_external_penalty_function,
    squared_penalty,
    total_violation,
)
```

І додай ці назви в `__all__`:

```python
"circle_constraint",
"make_external_penalty_function",
"squared_penalty",
"total_violation",
```

---

## 4. Оновити `optimization/experiments.py`

### 4.1. Додати імпорт

У блоці імпортів додай:

```python
from .penalty import (
    circle_constraint,
    make_external_penalty_function,
    squared_penalty,
    total_violation,
)
```

Якщо у тебе там є окремий блок для запуску без пакета, тоді в нього також додай:

```python
from optimization.penalty import (
    circle_constraint,
    make_external_penalty_function,
    squared_penalty,
    total_violation,
)
```

---

### 4.2. Додати назви колонок

У `DISPLAY_COLUMN_LABELS` додай:

```python
"r": "коефіцієнт штрафу r",
"f_original": "значення початкової функції",
"F_penalty": "значення штрафної функції",
"constraint_value": "значення обмеження g(x)",
"violation": "порушення обмеження",
```

---

### 4.3. Додати функцію експерименту

У кінець `experiments.py`, але перед `run_experiments`, додай:

```python
def penalty_experiment(
    method_fn,
    base_params=None,
    r_values=(1, 10, 100, 1000, 10000),
    x_start=X_START,
):
    """
    Експеримент для методу штрафних функцій.

    На кожному етапі мінімізується:
    F(x, r) = f(x) + r * P(x)

    Після кожного значення r наступний запуск стартує з попередньої знайденої точки.
    """
    base_params = dict(BASE_PARAMS if base_params is None else base_params)
    constraints = [circle_constraint]

    rows = []
    x_current = np.asarray(x_start, dtype=float).reshape(-1)

    for r in r_values:
        penalty_f = make_external_penalty_function(
            f=power_function,
            constraints=constraints,
            r=r,
        )

        result = method_fn(
            penalty_f,
            x_current,
            **base_params,
        )

        x_final = np.asarray(result["x_final"], dtype=float).reshape(-1)
        g_value = circle_constraint(x_final)
        violation = total_violation(x_final, constraints)

        rows.append(
            {
                "r": r,
                "x_final": format_point(x_final),
                "f_original": float(power_function(x_final)),
                "F_penalty": float(result["f_final"]),
                "constraint_value": float(g_value),
                "violation": float(violation),
                "iterations": int(result["iterations"]),
                "func_calls": int(result["func_calls"]),
                "status": STATUS_LABELS.get(
                    result.get("status", "unknown"),
                    result.get("status", "unknown"),
                ),
            }
        )

        # Наступний етап методу зовнішньої точки стартує з попереднього результату.
        x_current = x_final

    return pd.DataFrame(rows)
```

---

### 4.4. Додати порівняння МНС і ПАРТАН-МНС для штрафної функції

Після `penalty_experiment` додай:

```python
def compare_penalty_methods(base_params=None, r_values=(1, 10, 100, 1000, 10000)):
    base_params = dict(BASE_PARAMS if base_params is None else base_params)

    return {
        "МНС": penalty_experiment(
            method_fn=steepest_descent,
            base_params=base_params,
            r_values=r_values,
        ),
        "ПАРТАН-МНС": penalty_experiment(
            method_fn=partan_steepest_descent,
            base_params=base_params,
            r_values=r_values,
        ),
    }
```

---

## 5. Код для ноутбука після траєкторій МНС і ПАРТАН-МНС

Після блоку, де в тебе вже побудовані траєкторії пошуку, додай новий розділ.

Markdown-комірка:

```markdown
## Метод штрафних функцій для умовної оптимізації

Для умовної оптимізації використовується метод штрафних функцій, а саме метод зовнішньої точки. За основу взято степеневу функцію, яка вже використовувалась у задачі безумовної оптимізації.

Допустима область задається обмеженням:

$$x_1^2 + x_2^2 \le 1$$

Ця область є випуклою. Безумовний мінімум функції знаходиться в точці $$(1;1)$$, яка не належить допустимій області, оскільки:

$$1^2 + 1^2 = 2 > 1$$

Тому задача відповідає випадку розташування локального мінімуму поза випуклою допустимою областю.

Штрафна функція має вигляд:

$$F(x,r)=f(x)+r\max(0, x_1^2+x_2^2-1)^2$$
```

Python-комірка:

```python
from optimization.experiments import compare_penalty_methods, DISPLAY_COLUMN_LABELS

penalty_tables = compare_penalty_methods()
```

Python-комірка для МНС:

```python
penalty_tables["МНС"].rename(columns=DISPLAY_COLUMN_LABELS)
```

Python-комірка для ПАРТАН-МНС:

```python
penalty_tables["ПАРТАН-МНС"].rename(columns=DISPLAY_COLUMN_LABELS)
```

---

## 6. Що показувати в таблиці

Для кожного значення `r` треба показати:

```text
r
кінцева точка
значення початкової функції f(x)
значення штрафної функції F(x, r)
значення обмеження g(x)
порушення обмеження
кількість ітерацій
кількість викликів функції
статус
```

Головне, на що треба дивитися:

```text
constraint_value <= 0
violation -> 0
```

Якщо `violation` зменшується або стає дуже малим, то метод штрафних функцій працює нормально.

---

## 7. Короткий текст для звіту

Можна вставити в роботу такий текст:

```text
Для розв’язання задачі умовної оптимізації було використано метод штрафних функцій, а саме метод зовнішньої точки. Як цільову функцію використано степеневу функцію, досліджену в попередньому розділі. Допустиму область задано обмеженням x1^2 + x2^2 <= 1, яке формує випуклу область.

Безумовний мінімум степеневої функції знаходиться в точці (1; 1). Однак ця точка не належить допустимій області, оскільки 1^2 + 1^2 = 2 > 1. Отже, задача відповідає випадку, коли локальний мінімум розташований поза випуклою допустимою областю.

Для врахування обмеження було побудовано штрафну функцію F(x, r) = f(x) + r * max(0, x1^2 + x2^2 - 1)^2. Далі для різних значень коефіцієнта штрафу r було виконано мінімізацію штрафної функції методами МНС та ПАРТАН-МНС. Отримані результати дозволяють оцінити, як збільшення коефіцієнта штрафу впливає на наближення розв’язку до допустимої області.
```

---

## 8. Додаткові правки, які бажано зробити

### 8.1. `gradients.py`

Зараз у `numerical_gradient` значення `f_x = f(x)` рахується навіть для центральної схеми, хоча там воно не потрібне. Це зайвий виклик функції і може псувати підрахунок кількості викликів.

Було:

```python
f_x = float(f(x_arr))
```

Краще так:

```python
f_x = None
if scheme in {"forward", "backward"}:
    f_x = float(f(x_arr))
```

І далі:

```python
if scheme == "forward":
    grad[i] = (float(f(x_arr + step)) - f_x) / h
elif scheme == "backward":
    grad[i] = (f_x - float(f(x_arr - step))) / h
else:
    grad[i] = (float(f(x_arr + step)) - float(f(x_arr - step))) / (2.0 * h)
```

---

### 8.2. `line_search.py` і параметр Свена

Зараз `sven_alpha` фактично використовується як готовий крок для Свена. Якщо робити ближче до методички, то треба рахувати:

```text
Δλ = α * ||x_k|| / ||s_k||
```

Тобто в `steepest_descent.py` і `partan_steepest_descent.py` перед викликом Свена можна зробити так:

```python
s_norm = max(float(np.linalg.norm(s_k)), 1e-12)
x_norm = max(float(np.linalg.norm(xk)), 1e-12)
sven_delta = sven_alpha * x_norm / s_norm

a, b = sven_interval(phi, alpha=sven_delta)
```

Назву параметра `sven_alpha` можна залишити, але в коді він має використовуватись саме для обчислення `sven_delta`.

---

## 9. Мінімальний порядок виконання

1. Додати `optimization/penalty.py`.
2. Оновити `optimization/__init__.py`.
3. Оновити імпорти в `experiments.py`.
4. Додати `penalty_experiment`.
5. Додати `compare_penalty_methods`.
6. У ноутбуку після траєкторій додати блок про метод штрафних функцій.
7. Побудувати таблиці для МНС і ПАРТАН-МНС.
8. У висновку коротко порівняти, який метод дав менше викликів функції і як змінювалось порушення обмеження.
