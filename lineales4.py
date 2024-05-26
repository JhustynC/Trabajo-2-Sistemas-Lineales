import numpy as np
from scipy.signal import lfilter, convolve
import matplotlib.pyplot as plt

def suma_de_concolucion(x, h):
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)
    
    for n in range(N + M - 1):
        y[n] = sum(x[k] * h[n - k] for k in range(max(0, n - M + 1), min(n + 1, N)))
    return y

# Prueba de la función de convolución lineal
x = np.array([0, 1, 2, 3, 2, 1, 0], dtype=float)
h = np.array([1, 1, 1, 1, 1, 1], dtype=float)

# Convolución implementada
y_custom = suma_de_concolucion(x, h)
print("Resultado de la convolución implementada:", y_custom)

# Convolución con scipy.signal.convolve
y_convolve = convolve(x, h)
print("Resultado de scipy.signal.convolve:", y_convolve)

# Convolución con scipy.signal.lfilter
y_filter = lfilter(h, 1, np.pad(x, (0, len(h)-1)))
print("Resultado de scipy.signal.lfilter:", y_filter)

# Graficar las señales
plt.figure(figsize=(5, 6))

# Señal x
plt.subplot(3, 1, 1)
plt.stem(np.arange(len(x)), x)
plt.title('Señal x')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid()

# Señal h
plt.subplot(3, 1, 2)
plt.stem(np.arange(len(h)), h)
plt.title('Señal h')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid()

# Resultado de la convolución
plt.subplot(3, 1, 3)
#plt.stem(np.arange(len(y_custom)), y_custom, label='Implementación personalizada', linefmt='C0-', markerfmt='C0o', basefmt='C0-')
#plt.stem(np.arange(len(y_convolve)), y_convolve, label='scipy.signal.convolve', linefmt='C1--', markerfmt='C1s', basefmt='C1--')
plt.stem(np.arange(len(y_filter)), y_filter, label='scipy.signal.lfilter', linefmt='C2-.', markerfmt='C2^', basefmt='C2-.')
plt.title('Resultado de la convolución')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
#plt.legend()
plt.grid()

plt.tight_layout()
plt.show()