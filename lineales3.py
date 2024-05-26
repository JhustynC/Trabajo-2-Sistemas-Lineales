import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Generar los valores de n
n = np.arange(-10, 11, 1)

# Definir la función u[n]
#u = lambda n: 1 if n >= 0 else 0
u = lambda n: np.where(n >= 0, 1, 0)

# Definir la función f[n]
f_n = lambda n: np.exp(-n/5) * np.cos((n * np.pi) / 5) * u(n)

# Calcular f[n] para los valores de n
f_n_values = f_n(-2*n+1)

markerline, stemlines, baseline = plt.stem(n, f_n_values)
plt.setp(stemlines, 'linewidth', 2)  # Ajustar el grosor de los palitos
plt.setp(markerline, 'markersize', 6)  # Ajustar el tamaño de los marcadores


# Graficar f[n] usando stem
#plt.stem(n, f_n_values)
plt.title('$f[n] = e^{-n/5} \cdot \cos(\pi n/5) \cdot u[n]$')
plt.xlabel('n')
plt.ylabel('$f[n]$')
plt.xticks(np.arange(-10, 11, 1))
plt.grid(True)
plt.show()
