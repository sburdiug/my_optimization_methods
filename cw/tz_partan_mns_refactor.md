# ТЗ: рефакторинг МНС и ПАРТАН-МНС для курсовой

## 1. Цель

Переделать текущую реализацию метода найшвидшого спуску и ПАРТАН-МНС так, чтобы основное исследование выполнялось численно, а не через аналитические вычисления SymPy.

SymPy можно оставить только для задания функции и проверки формул, но не для основного вычисления градиента и шага `lambda`.

---

## 2. Основная функция исследования

Использовать степеневую функцию:

```python
def power_function(x):
    x1, x2 = x
    return (10 * (x1 - x2)**2 + (x1 - 1)**2)**4
```

Начальная точка:

```python
x_start = [-1.2, 0.0]
```

Ожидаемый минимум:

```python
x_min = [1.0, 1.0]
f_min = 0.0
```

---

## 3. Что убрать из основной реализации

В основных методах оптимизации не использовать:

```python
sp.diff(...)
sp.solve(...)
sp.hessian(...)
```

То есть убрать аналитическое вычисление:

- градиента через `sp.diff`;
- оптимального шага через производную `dphi/dlambda`;
- точного шага через Hessian.

---

## 4. Структура файлов

Желательная структура:

```text
optimization/
│
├── functions.py
├── gradients.py
├── line_search.py
├── steepest_descent.py
├── partan_steepest_descent.py
├── experiments.py
└── plots.py
```

---

## 5. Файл `gradients.py`

Сделать отдельную функцию численного градиента:

```python
def numerical_gradient(f, x, h=1e-4, scheme="central"):
    ...
```

Поддержать схемы:

```python
scheme="forward"   # правая разность
scheme="backward"  # левая разность
scheme="central"   # центральная разность
```

Формулы:

```python
forward:  (f(x + h * e_i) - f(x)) / h
backward: (f(x) - f(x - h * e_i)) / h
central:  (f(x + h * e_i) - f(x - h * e_i)) / (2 * h)
```

Параметр `h` должен передаваться снаружи, чтобы можно было исследовать его влияние.

---

## 6. Файл `line_search.py`

Сделать методы одномерного поиска:

```python
def sven_interval(phi, alpha=0.01):
    ...
```

```python
def golden_section_search(phi, a, b, eps=1e-3):
    ...
```

```python
def dsk_powell_search(phi, a, b, eps=1e-3):
    ...
```

`phi` — одномерная функция:

```python
phi = lambda lam: f(xk + lam * s_k)
```

Параметры `alpha` и `eps` должны передаваться снаружи.

---

## 7. Файл `steepest_descent.py`

Метод найшвидшого спуску должен принимать параметры:

```python
def steepest_descent(
    f,
    x_start,
    max_iter=1000,
    eps=1e-4,
    derivative_h=1e-4,
    gradient_scheme="central",
    line_search_method="golden",
    line_search_eps=1e-3,
    sven_alpha=0.01,
    stop_criterion="combined",
):
    ...
```

На каждой итерации:

```python
grad_k = numerical_gradient(f, xk, h=derivative_h, scheme=gradient_scheme)
s_k = -grad_k
phi = lambda lam: f(xk + lam * s_k)
a, b = sven_interval(phi, alpha=sven_alpha)
lambda_opt = выбранный_метод_одномерного_поиска(phi, a, b, eps=line_search_eps)
x_next = xk + lambda_opt * s_k
```

---

## 8. Файл `partan_steepest_descent.py`

ПАРТАН-МНС должен принимать те же параметры:

```python
def partan_steepest_descent(
    f,
    x_start,
    max_iter=1000,
    eps=1e-4,
    derivative_h=1e-4,
    gradient_scheme="central",
    line_search_method="golden",
    line_search_eps=1e-3,
    sven_alpha=0.01,
    stop_criterion="combined",
):
    ...
```

Логика направления:

```python
if k < 2:
    method_name = "mns"
    s_k = -grad_k
else:
    if k % 2 == 0:
        method_name = "partan"
        s_k = xk - points[-3]
    else:
        method_name = "mns"
        s_k = -grad_k
```

Для каждого направления шаг `lambda` искать только через одномерный поиск:

```python
phi = lambda lam: f(xk + lam * s_k)
a, b = sven_interval(phi, alpha=sven_alpha)
lambda_opt = golden_section_search(...) или dsk_powell_search(...)
```

---

## 9. Подсчет вызовов функции

Добавить счетчик вызовов целевой функции.

Вариант реализации:

```python
class FunctionCounter:
    def __init__(self, f):
        self.f = f
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.f(x)
```

В результатах каждого запуска сохранять:

```python
func_calls
```

Это главный показатель для таблиц исследования.

---

## 10. Критерии остановки

Поддержать минимум два варианта:

```python
stop_criterion="gradient"
```

Остановка по норме градиента:

```python
||grad f(xk)|| <= eps
```

```python
stop_criterion="combined"
```

Комбинированная остановка:

```python
||x_next - xk|| / ||xk|| <= eps
abs(f(x_next) - f(xk)) <= eps
```

---

## 11. Что возвращать из методов

Оба метода должны возвращать словарь:

```python
{
    "method": "steepest_descent" или "partan_steepest_descent",
    "x_final": ...,
    "f_final": ...,
    "grad_final": ...,
    "grad_norm_final": ...,
    "iterations": ...,
    "func_calls": ...,
    "points": ...,
    "history": ...,
    "params": {...},
}
```

В `history` сохранять для каждой итерации:

```python
k
method_name
x
f_x
grad
grad_norm
s
lambda_opt
x_next
f_next
func_calls
```

---

## 12. Файл `experiments.py`

Сделать набор экспериментов для таблиц курсовой.

### 12.1. Влияние шага `h`

```python
h_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
```

Фиксировать остальные параметры.

### 12.2. Влияние схемы производных

```python
schemes = ["forward", "backward", "central"]
```

### 12.3. Влияние метода одномерного поиска

```python
line_search_methods = ["golden", "dsk_powell"]
```

### 12.4. Влияние точности одномерного поиска

```python
line_search_eps_values = [1e-1, 1e-2, 1e-3, 1e-4]
```

### 12.5. Влияние параметра Свена

```python
sven_alpha_values = [0.001, 0.005, 0.01, 0.05, 0.1]
```

### 12.6. Влияние критерия остановки

```python
stop_criteria = ["gradient", "combined"]
```

### 12.7. Сравнение МНС и ПАРТАН-МНС

Запустить оба метода с одинаковыми параметрами и сравнить:

- финальную точку;
- значение функции;
- количество итераций;
- количество вызовов функции;
- траекторию поиска.

---

## 13. Формат таблиц результатов

Для каждого эксперимента сохранять таблицу с колонками:

```text
parameter_value | x_final | f_final | iterations | func_calls | status
```

Для сравнения методов:

```text
method | x_final | f_final | iterations | func_calls
```

---

## 14. Графики

Сделать минимум:

1. Траектория поиска МНС.
2. Траектория поиска ПАРТАН-МНС.
3. График зависимости количества вызовов функции от исследуемого параметра.
4. Сравнение МНС и ПАРТАН-МНС по количеству вызовов функции.

---

## 15. Критерий готовности

Работа считается готовой, если:

- в основных методах нет `sp.diff`, `sp.solve`, `sp.hessian`;
- градиент считается через `numerical_gradient`;
- шаг `lambda` считается через золотое сечение или ДСК-Пауэлла;
- можно менять `h`, схему производных, метод одномерного поиска, точность поиска, параметр Свена и критерий остановки;
- считается количество вызовов целевой функции;
- результаты можно вывести в таблицы;
- строится траектория поиска для МНС и ПАРТАН-МНС.
