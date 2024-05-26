import numpy as np
import matplotlib.pyplot as plt

# Condiciones iniciales
y = [1, 2]  # y[-2] = 1, y[-1] = 2
x = []
n_values = 10  # Número de puntos a calcular

# Generar x[n]
x = [n if n >= 0 else 0 for n in range(n_values + 2)]

# Calcular y[n] iterativamente
for n in range(n_values):
    y_n2 = y[-1] - (0.24 * y[-2]) + x[n + 2] - (2 * x[n + 1])
    y.append(y_n2)

# Truncar y para que tenga el mismo tamaño que x
y = y[2:]  # Eliminar los dos primeros elementos que corresponden a y[-2] y y[-1]

# Graficar
plt.figure(figsize=(12, 6))

# Graficar x[n]
plt.subplot(1, 2, 1)
plt.stem(range(n_values), x[:n_values], use_line_collection=True)
plt.title('Entrada x[n]')
plt.xlabel('n')
plt.ylabel('x[n]')

# Graficar y[n]
plt.subplot(1, 2, 2)
plt.stem(range(n_values), y[:n_values], use_line_collection=True)
plt.title('Salida y[n]')
plt.xlabel('n')
plt.ylabel('y[n]')

plt.tight_layout()
plt.show()