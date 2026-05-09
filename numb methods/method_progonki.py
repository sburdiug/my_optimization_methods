
def p_forward(a, b, c, d):

    n = len(b)
    alph = [0.0] * n
    gamm = [0.0] * n


    # Початкові значення
    alph[0] = c[0] / b[0]
    gamm[0] = -d[0] / b[0]


    for i in range(1, n):
        alph[i] = c[i] /  (b[i] - a[i] * alph[i - 1])
        gamm[i] = (a[i] * gamm[i - 1] - d[i]) /  (b[i] - a[i] * alph[i - 1])

    return alph, gamm

def p_backward(alph, gamm):

    n = len(gamm)
    x = [0.0] * n

    x[-1] = gamm[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alph[i] * x[i + 1] + gamm[i]
    return x


ai = [0, 2.55, 1.59, 2.29]
bi = [-4.28, -6.51, -7.79, -6.24]
ci = [-1.49, -0.85, -2.58, 0]
di = [1.63, 3.91, 2.54, 5.52]





alpha, gamma = p_forward(ai,bi,ci,di)
print("\nВектори прогоночних коефіцієнтів")
print("i | αi  |  γi ")
for i in range(len(alpha)-1):
    print(f"{i + 1} | {round(alpha[i], 4)} | {round(gamma[i],4)}")

xi = p_backward(alpha,gamma)
print("\nВектор розв’язків")
print("i | xi ")
for i in range(len(xi)):
    print(f"{i + 1} | {round(xi[i], 4)}")

