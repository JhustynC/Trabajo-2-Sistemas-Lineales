import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# INGRESO
muestras = 10

# condiciones iniciales ascendente ...,y[-2],y[-1]
y0 = [1,2]

# PROCEDIMIENTO
m0 = len(y0)
m  = muestras + m0

# vectores
n  = np.arange(-m0,muestras,1)
xi = np.zeros(m, dtype=float)
yi = np.zeros(m, dtype=float)

# Añade condiciones iniciales
xi[0:m0] = 0
yi[0:m0] = y0

# Calcula los siguientes valores
for k in range(m0,m,1):
    xi[k] = n[k]
    yi[k] = yi[k-1]-0.24*yi[k-2]+xi[k]-2*xi[k-1]

# tabla de valores
# Concatenamos las filas y transponemos la tabla
tabla = np.vstack((n, xi, yi)).T

# Creamos un DataFrame usando pandas
df = pd.DataFrame(tabla, columns=['n', 'xi', 'yi'])

# Configuramos la impresión para mostrar dos decimales
pd.set_option('display.precision', 2)

# Imprimimos el resultado
print('muestras:', len(n))
print(df.to_string(index=False))

# Graficamos los resultados
plt.figure(figsize=(10, 5))


# Gráfica
plt.subplot(1,2,1)
plt.stem(n,xi)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid()
plt.title('Entrada x[n]=n*u[n]')
plt.xlabel('n')
plt.ylabel('x[n]')


plt.subplot(1,2,2)
plt.stem(n,yi)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid()
plt.title('Respuesta y[n]=y[n-1]-0.24*y[n-2]+x[n]-2*x[n-1]')


plt.show()
