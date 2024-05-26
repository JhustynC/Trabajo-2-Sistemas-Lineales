import numpy as np
import matplotlib.pyplot as plt

u_n = lambda n: np.where(n >= 0, 1, 0)

# Definir las funciones x(t) y h(t)
def x(t):
    return np.exp(2 * t) * u_n(-t)

def h(t):
    return u_n(t-3)

# Crear un rango de valores de t
t = np.linspace(-5, 10, 10000)
dt = t[1] - t[0]

# Evaluar las funciones x(t) y h(t)
x_t = x(t)
h_t = h(t)

# Calcular la convolución usando numpy
y_t = np.convolve(h_t, x_t, mode='same') * dt  # El factor (t[1] - t[0]) es para la corrección de la escala

# Graficar las funciones y la convolución
plt.figure(figsize=(10, 3.5))

plt.subplot(1, 3, 1)
plt.plot(t, x_t, label='x(t)', color='blue')
plt.title('x(t)')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(t, h_t, label='h(t)', color='green')
plt.title('h(t)')
plt.xlabel('t')
plt.ylabel('h(t)')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(t, y_t, label='y(t) = x(t) * h(t)', color='red')
plt.title('Convolución y(t)')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
