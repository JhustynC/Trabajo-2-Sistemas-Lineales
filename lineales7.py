import numpy as np
import matplotlib.pyplot as plt


# Definimos la función escalón unitario
def unit_step(t):
    return np.heaviside(t, 1)

# Definimos las funciones x(t) y h(t)
def x(t):
    return np.exp(2*t) * unit_step(-t)

def h(t):
    return unit_step(t-3)

# Definimos los parámetros de tiempo
t = np.linspace(0, 10, 1000)  # Intervalo de tiempo de 0 a 10 con 1000 puntos
dt = t[1] - t[0]

# Calculamos las funciones en el intervalo de tiempo
x_t = x(t)
h_t = h(t)

# Calculamos la convolución de x(t) y h(t)
y_t = np.convolve(x_t, h_t) * dt
t_conv = np.linspace(0, 2 * t[-1], len(y_t))  # Intervalo de tiempo para la convolución

# Graficamos las funciones y la convolución
plt.figure(figsize=(12, 3))

plt.subplot(1, 3, 1)
plt.plot(t, x_t, label='$x(t) = e^{-t}u(t)$', color='blue')
plt.title('$x(t)$')
plt.xlabel('$t$')
plt.ylabel('$x(t)$')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(t, h_t, label='$h(t) = e^{-2t}u(t)$', color='green')
plt.title('$h(t)$')
plt.xlabel('$t$')
plt.ylabel('$h(t)$')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(t_conv, y_t, label='$y(t) = (x * h)(t)$', color='red')
plt.title('Convolución $y(t)$')
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

