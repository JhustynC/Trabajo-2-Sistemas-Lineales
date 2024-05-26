import numpy as np
import matplotlib.pyplot as plt


# Definimos la función escalón unitario
def unit_step(t):
    return np.heaviside(t, 1)

# Definimos las funciones x(t) y h(t)
def x(t):
    return np.exp(-t) * unit_step(t)

def h(t):
    return np.exp(-2*t) * unit_step(t)

# Definimos los parámetros de tiempo
t = np.linspace(-5, 10, 1000)  # Intervalo de tiempo de 0 a 10 con 1000 puntos
dt = t[1] - t[0]

# Calculamos las funciones en el intervalo de tiempo
x_t = x(t)
h_t = h(t)

# Calculamos la convolución de x(t) y h(t)
# se multiplica por dt para obtener la escala correcta.
y_t = np.convolve(h_t, x_t, mode='same') * dt  

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


