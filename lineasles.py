import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Definimos los parámetros
n_max = 10 # Numero de muestras
n = np.arange(0, n_max+1) #Vector de muestras
x = n**2 # Definimos la entrada x[n] 

# # Inicializamos un array con ceros 
# y asignamos la condición inicial y[-1]
y = np.zeros_like(n, dtype=float) 
y_1 = 16
#! 16 es un valor pasdo por lo cual podemos 
#! sacar valores despues de -1 es decir
#! 0,1,2,3,4,..

# Realizamos la iteración
for i in range(n_max + 1):
    if i == 0:
        y[i] = x[i] + (0.5 * y_1)
    else:
        y[i] = x[i] + 0.5 * y[i-1]


# Concatenamos las filas y transponemos la tabla
tabla = np.vstack((n, x, y)).T

# Creamos un DataFrame usando pandas
df = pd.DataFrame(tabla, columns=['n', 'xi', 'yi'])

# Configuramos la impresión para mostrar dos decimales
pd.set_option('display.precision', 2)

# Imprimimos el resultado
print('muestras:', len(n))
print(df.to_string(index=False))

# Graficamos los resultados
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1) #Para posicionar el grafico
plt.stem(n, x)#Dibujar puntos y rectas al eje
plt.title('Entrada x[n] = n^2 * u[n]')
plt.xlabel('n')#Titulo del eje x
plt.ylabel('x[n]')#Titulo del eje y
plt.grid()#Mostrar el fonde del plano

plt.subplot(1, 2, 2)#Para posicionar el grafico
plt.stem(n, y)
plt.title('Respuesta y[n]')
plt.xlabel('n')#Titulo del eje x
plt.ylabel('y[n]')#Titulo del eje y
plt.grid()#Mostrar el fonde del plano

plt.tight_layout()
plt.show()#Mostrar la figura con los graficos
