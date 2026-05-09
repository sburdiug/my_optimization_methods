def output_matrix(matrix):
    for row in matrix:
        print([round(x, 6) for x in row])  # округлення для зручності
    print('')


def into_LU(A):
    """
    Схема Холєцького для факторизації A = L * U
    L — нижня трикутна, U — верхня трикутна
    A — квадратна матриця
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        U[i][i] = 1.0  # одинична діагональ

    for j in range(n):
        # d_jj
        s = 0
        for k in range(j):
            s += L[j][k] * U[k][j]
        L[j][j] = A[j][j] - s

        # елементи стовпця L нижче діагоналі (i>j)
        for i in range(j + 1, n):
            s = 0
            for k in range(j):
                s += L[i][k] * U[k][j]
            L[i][j] = A[i][j] - s

        # елементи рядка U праворуч від діагоналі (i<j → j>i)
        for i in range(j + 1, n):
            s = 0
            for k in range(j):
                s += L[j][k] * U[k][i]
            U[j][i] = (A[j][i] - s) / L[j][j]

    return L, U


def det_LU(L, U):
    det = 1.0
    for i in range(len(L)):
        det *= L[i][i]
    return det


myMatrix = [[10.79, 3.41, 5.33, 2.47],
            [5.1, 10.99, 10.11, 3.2],
            [4.91, 9.39, 10.45, 2.96],
            [2.89, 4.76, 2.18, 9.1]]

print("Матриця A:")
output_matrix(myMatrix)

matrixL, matrixU = into_LU(myMatrix)

print("Матриця L:")
for row in matrixL:
    print([round(x, 4) for x in row])

print("\nМатриця U:")
for row in matrixU:
    print([round(x, 4) for x in row])

detA = det_LU(matrixL, matrixU)
print(f"Визначник det(A) = {detA:.4f}")