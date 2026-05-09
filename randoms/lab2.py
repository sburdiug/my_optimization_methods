import numpy as np
import matplotlib.pyplot as plt


t = np.arange(0, 4.1, 0.1)
t_prime = np.arange(0, 4.1, 0.1)

# Створюємо сітку
T, T_prime = np.meshgrid(t, t_prime)

# Обчислюємо значення кореляційної функції
K_x = (4/3) * np.cos(0.6 * T) * np.cos(0.6 * T_prime)

# Побудова 3D графіка
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T, T_prime, K_x, cmap='viridis')

ax.view_init(elev=30, azim=135)
ax.set_box_aspect((1,1,0.6))

ax.set_title("Графік кореляційної функції K_x(t, t')")
ax.set_xlabel("t")
ax.set_ylabel("t'")
ax.set_zlabel("K_x(t, t')")
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()