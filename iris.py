import numpy as np
import pandas as pd
import os
from os import system

system("cls")

# Leer iris.data con ruta relativa
ruta_archivo = os.path.join(os.path.dirname(__file__), "IRIS", "iris.data")

columnas = ["LS", "AS", "LP", "AP", "Clase"]
df = pd.read_csv(ruta_archivo, header=None, names=columnas) # data frmae :)
df = df.dropna()

# Entrenamiento del perceptrón con detección de convergencia
def entrenamiento(X, y, lr=0.1, epocas=100):
    X = np.hstack([X, np.ones((X.shape[0], 1))])  # añadir bias
    w = np.zeros(X.shape[1])
    convergio = False
    epoca_real = 0  # contador real de épocas

    for epoca in range(epocas):
        errores = 0
        for xi, yi in zip(X, y):
            y_pred = np.sign(np.dot(w, xi))
            if y_pred == 0:
                y_pred = -1
            if yi != y_pred:
                errores += 1
                w += lr * (yi - y_pred) * xi
        epoca_real = epoca + 1
        if errores == 0:
            convergio = True
            break

    return w, convergio, epoca_real

# Menú de opciones
print("Selecciona las clases para entrenar el Perceptrón:\n")
print("1.- Setosa - Versicolor")
print("2.- Setosa - Virginica")
print("3.- Versicolor - Virginica\n")
op = int(input("Opción: "))

if op == 1:
    clase1, clase2 = "Iris-setosa", "Iris-versicolor"
    nombre = "Setosa-Versicolor"
elif op == 2:
    clase1, clase2 = "Iris-setosa", "Iris-virginica"
    nombre = "Setosa-Virginica"
elif op == 3:
    clase1, clase2 = "Iris-versicolor", "Iris-virginica"
    nombre = "Versicolor-Virginica"
else:
    print("Opción NO válida.")
    exit()

# Filtro de datos
df_sub = df[(df["Clase"] == clase1) | (df["Clase"] == clase2)].copy()
X = df_sub[["LS", "AS", "LP", "AP"]].values
y = np.where(df_sub["Clase"] == clase1, -1, 1)

# Entrenamiento
w, convergio, epoca_real = entrenamiento(X, y, lr=0.1, epocas=51)
X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
y_cal = np.sign(np.dot(X_bias, w))

# Convergencia
if convergio:
    estado = f"Parámetros encontrados."
else:
    estado = f"Parámetros ÑO encontrados."

print(f"\n{estado}")
print("Epocas: ",epoca_real,"\n")

print("Pesos:")
for i in range(len(w) - 1):
    print(f"w({i+1}) = {w[i]:.4f}")
print(f"Umbral = {w[-1]:.4f}\n")

# Mostrar resultados
print(f"Perceptrón entrenado: {nombre}\n")
print("      LS      AS      LP      AP      Y     Ycal")
for i in range(min(100, len(X))):
    print(f"{i+1:2d}.- {X[i,0]:<6.2f}  {X[i,1]:<6.2f}  {X[i,2]:<6.2f}  {X[i,3]:<6.2f}   {int(y[i]):>2d}    {y_cal[i]:>4.1f}")

print("\nPesos finales del perceptrón: ", w, "\n")
