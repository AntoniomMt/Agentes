import numpy as np
import pandas as pd
import os
from os import system

system("cls")

# Leer iris.data con ruta relativa
ruta_archivo = os.path.join(os.path.dirname(__file__), "IRIS", "iris.data")

columnas = ["LS", "AS", "LP", "AP", "Clase"]
df = pd.read_csv(ruta_archivo, header=None, names=columnas)
df = df.dropna()

# Preparar datos para clasificación multiclase
X = df[["LS", "AS", "LP", "AP"]].values.astype("float32")

# Codificación One-Hot de las clases
clases_unicas = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
y_labels = df["Clase"].values

# Crear matriz One-Hot (150 muestras x 3 clases)
y = np.zeros((len(y_labels), 3), dtype="float32")
for i, label in enumerate(y_labels):
    if label == "Iris-setosa":
        y[i] = [1, 0, 0]
    elif label == "Iris-versicolor":
        y[i] = [0, 1, 0]
    elif label == "Iris-virginica":
        y[i] = [0, 0, 1]

# Importar TensorFlow y Keras
from tensorflow import random
from tensorflow.keras import Input, Model, Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# Fijar semilla para reproducibilidad
np.random.seed(32)

# Definir modelo de Red Neuronal para 3 clases
model = Sequential()
model.add(Dense(80, activation='relu', input_dim=4))   # 4 características de entrada
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))              # 3 salidas (una por clase)

# Compilar el modelo
model.compile(
    loss='categorical_crossentropy',  # Para multiclase
    optimizer='adam', 
    metrics=['accuracy']
)

# Entrenar el modelo
print("Entrenando Red Neuronal para clasificación de 3 especies de Iris")
print("=" * 80)
history = model.fit(X, y, epochs=350, verbose=0)

# Evaluar el modelo
loss, accuracy = model.evaluate(X, y, verbose=0)

print(f"\nResultados del entrenamiento:")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%\n")

# Hacer predicciones
y_predicted = model.predict(X, verbose=0)

# Convertir predicciones a clases (el índice con mayor probabilidad)
y_pred_class = np.argmax(y_predicted, axis=1)
y_true_class = np.argmax(y, axis=1)

# Nombres de las clases para visualización
nombres_clases = ["Setosa", "Versicolor", "Virginica"]

# Mostrar resultados
print("Resultados de la clasificación:\n")
print("      LS      AS      LP      AP    |  Esperada  |    Calculada (Probabilidades)      | Clase")
print("-" * 110)

for i in range(len(X)):
    # Formatear salida esperada
    esperada_str = f"[{int(y[i,0])},{int(y[i,1])},{int(y[i,2])}]"
    
    # Formatear salida calculada con probabilidades
    calculada_str = f"[{y_predicted[i,0]:.2f},{y_predicted[i,1]:.2f},{y_predicted[i,2]:.2f}]"
    
    # Determinar si la predicción es correcta
    correcto = "✓" if y_pred_class[i] == y_true_class[i] else "✗"
    
    # Nombre de la clase predicha
    clase_predicha = nombres_clases[y_pred_class[i]]
    
    print(f"{i+1:3d}.- {X[i,0]:<6.2f}  {X[i,1]:<6.2f}  {X[i,2]:<6.2f}  {X[i,3]:<6.2f} | {esperada_str:>9s} | {calculada_str:>25s}  | {correcto} {clase_predicha}")

# Calcular matriz de confusión manual
print("\n" + "=" * 80)
print("RESUMEN DE CLASIFICACIÓN:\n")

for i, clase in enumerate(nombres_clases):
    correctos = np.sum((y_true_class == i) & (y_pred_class == i))
    total = np.sum(y_true_class == i)
    print(f"{clase:12s}: {correctos}/{total} correctas ({correctos/total*100:.1f}%)")

# Total de errores
errores = np.sum(y_pred_class != y_true_class)
print(f"\nTotal de errores: {errores}/{len(y)}")
print(f"Precisión global: {(1 - errores/len(y))*100:.2f}%\n")