def output_matrix(matrix):
    for row in matrix:
        print([round(x, 6) for x in row])  # округлення для зручності
    print('')


def gauss_solve(A, b):
    """Розв'язання Ax = b методом Гаусса зі зворотним ходом."""
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)] # Формуємо розширену матрицю [A|b]

    # === Прямий хід ===
    # Зводимо матрицю до верхньотрикутного вигляду
    for i in range(n):
        pivot = M[i][i]


        if pivot == 0:                   # Якщо піводіагональний елемент = 0 міняємо рядки
            for k in range(i+1, n):
                if M[k][i] != 0:
                    M[i], M[k] = M[k], M[i]
                    pivot = M[i][i]
                    break
            else:
                raise ValueError("вироджена СЛАР")

        for j in range(i, n+1):         # Нормалізуємо рядок так, щоб піводіагональний елемент = 1
            M[i][j] /= pivot


        for k in range(i+1, n):         # Виключаємо невідомі з нижніх рядків
            factor = M[k][i]
            for j in range(i, n+1):
                M[k][j] -= factor * M[i][j]

    # === Зворотний хід ===
    # Обчислюємо невідомі x, починаючи знизу вгору
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        x[i] = round(M[i][n] - sum(M[i][j]*x[j] for j in range(i+1, n)), 2)
    return x


def inverse_matrix_gauss(matrix):
    """Обернена матриця через метод Гаусса"""
    n = len(matrix)
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  # Створюємо одиничну матрицю
    inv = []

    # Для кожного стовпця одиничної матриці розв'язуємо СЛАР
    for j in range(n):
        col = gauss_solve(matrix, [I[i][j] for i in range(n)])
        inv.append(col)

    # Транспонуємо список стовпців -> отримуємо матрицю
    return [list(row) for row in zip(*inv)]


def multiply_matrix(matrix1, matrix2):
    """Множення матриць"""
    n = len(matrix1)
    m = len(matrix2[0])
    p = len(matrix2)
    C = [[0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = sum(matrix1[i][k] * matrix2[k][j] for k in range(p))
    return C


def refine_inverse(matrix, invMatrix, eps=1e-3, max_iter=15):
    """Ітераційне уточнення оберненої матриці"""
    n = len(matrix)
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  # Створюємо одиничну матрицю
    D = invMatrix

    for _ in range(max_iter):
        # F = I - A*D
        AD = multiply_matrix(matrix, D)
        F = [[I[i][j] - AD[i][j] for j in range(n)] for i in range(n)]

        # D_new = D + D*F
        DF = multiply_matrix(D, F)
        D_new = [[D[i][j] + DF[i][j] for j in range(n)] for i in range(n)]

        # різниця між ітераціями
        diff = max(abs(D_new[i][j] - D[i][j]) for i in range(n) for j in range(n))
        if diff < eps:
            return D_new

        D = D_new
    return D


# ======= Тест =======
myMatrix = [[10.79, 3.41, 5.33, 2.47],
            [5.1, 10.99, 10.11, 3.2],
            [4.91, 9.39, 10.45, 2.96],
            [2.89, 4.76, 2.18, 9.1]]
myEps = 0.02

print("Матриця M:")
output_matrix(myMatrix)

invMatrix = inverse_matrix_gauss(myMatrix)
print("Початкове наближення M^-1(метод Гаусса):")
output_matrix(invMatrix)


print('Перевірка M * M^-1 ≈ I:')
test = multiply_matrix(myMatrix, invMatrix)
output_matrix(test)



inv_refined = refine_inverse(myMatrix, invMatrix, eps=myEps)
print("Уточнена матриця M^-1:")
output_matrix(inv_refined)

# Перевірка M * M^-1 ≈ I
print("Перевірка M * M^-1 ≈ I після уточнення:")
check = multiply_matrix(myMatrix, inv_refined)
output_matrix(check)
