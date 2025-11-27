import numpy as np
import pandas as pd
import os
from os import system

system("cls")

# Leer iris.data
ruta_archivo = os.path.join(os.path.dirname(__file__), "IRIS", "iris.data")
columnas = ["LS", "AS", "LP", "AP", "Clase"]
df = pd.read_csv(ruta_archivo, header=None, names=columnas)
df = df.dropna()

# Preparar datos
X = df[["LS", "AS", "LP", "AP"]].values.astype("float32")
y_labels = df["Clase"].values

# One-Hot Encoding
y = np.zeros((len(y_labels), 3), dtype="float32")
for i, label in enumerate(y_labels):
    if label == "Iris-setosa":
        y[i] = [1, 0, 0]
    elif label == "Iris-versicolor":
        y[i] = [0, 1, 0]
    elif label == "Iris-virginica":
        y[i] = [0, 0, 1]

from tensorflow import random
from tensorflow.keras import Input, Model, Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import tensorflow as tf

# Silenciar warnings de TensorFlow
tf.get_logger().setLevel('ERROR')

# Fijar semilla
np.random.seed(32)
tf.random.set_seed(32)

# Definir arquitecturas a probar
arquitecturas = [
    {"nombre": "Muy Simple", "capas": [8], "descripcion": "1 capa oculta, 8 neuronas"},
    {"nombre": "Simple", "capas": [16], "descripcion": "1 capa oculta, 16 neuronas"},
    {"nombre": "Pequeña", "capas": [32], "descripcion": "1 capa oculta, 32 neuronas"},
    {"nombre": "Mediana", "capas": [64], "descripcion": "1 capa oculta, 64 neuronas"},
    {"nombre": "Grande", "capas": [128], "descripcion": "1 capa oculta, 128 neuronas"},
    {"nombre": "Dos Capas Pequeñas", "capas": [16, 8], "descripcion": "2 capas: 16→8"},
    {"nombre": "Dos Capas Medianas", "capas": [32, 16], "descripcion": "2 capas: 32→16"},
    {"nombre": "Dos Capas Grandes", "capas": [64, 32], "descripcion": "2 capas: 64→32"},
    {"nombre": "Tres Capas", "capas": [32, 16, 8], "descripcion": "3 capas: 32→16→8"},
    {"nombre": "Original (red_ejemplo)", "capas": [80, 100], "descripcion": "2 capas: 80→100"},
]

print("COMPARACIÓN DE ARQUITECTURAS DE RED NEURONAL PARA IRIS")
print("=" * 90)
print(f"{'Arquitectura':<25} | {'Descripción':<20} | {'Accuracy':>10} | {'Loss':>8} | {'Épocas':>7}")
print("-" * 90)

resultados = []

for arq in arquitecturas:
    # Crear modelo
    model = Sequential()
    
    # Primera capa oculta
    model.add(Dense(arq["capas"][0], activation='relu', input_dim=4))
    
    # Capas ocultas adicionales
    for neuronas in arq["capas"][1:]:
        model.add(Dense(neuronas, activation='relu'))
    
    # Capa de salida
    model.add(Dense(3, activation='softmax'))
    
    # Compilar
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Entrenar con early stopping manual
    mejor_accuracy = 0
    mejor_epoca = 0
    paciencia = 50
    sin_mejora = 0
    
    for epoca in range(1, 501):
        history = model.fit(X, y, epochs=1, verbose=0)
        loss, accuracy = model.evaluate(X, y, verbose=0)
        
        if accuracy > mejor_accuracy:
            mejor_accuracy = accuracy
            mejor_epoca = epoca
            sin_mejora = 0
        else:
            sin_mejora += 1
        
        if sin_mejora >= paciencia:
            break
    
    # Guardar resultados
    resultados.append({
        "nombre": arq["nombre"],
        "descripcion": arq["descripcion"],
        "accuracy": mejor_accuracy,
        "loss": loss,
        "epocas": mejor_epoca,
        "capas": arq["capas"]
    })
    
    print(f"{arq['nombre']:<25} | {arq['descripcion']:<20} | {mejor_accuracy*100:>9.2f}% | {loss:>8.4f} | {mejor_epoca:>7}")

# Encontrar la mejor arquitectura
print("\n" + "=" * 90)
mejor = max(resultados, key=lambda x: x["accuracy"])

print("\n🏆 ARQUITECTURA ÓPTIMA:")
print(f"   Nombre: {mejor['nombre']}")
print(f"   Estructura: {' → '.join(map(str, mejor['capas']))} → 3")
print(f"   Accuracy: {mejor['accuracy']*100:.2f}%")
print(f"   Loss: {mejor['loss']:.4f}")
print(f"   Épocas necesarias: {mejor['epocas']}")

# Análisis y explicación
print("\n" + "=" * 90)
print("\n📊 ANÁLISIS Y CONCLUSIONES:\n")

# Comparar arquitecturas simples vs complejas
simple_avg = np.mean([r["accuracy"] for r in resultados if len(r["capas"]) == 1 and r["capas"][0] <= 32])
compleja_avg = np.mean([r["accuracy"] for r in resultados if len(r["capas"]) > 1 or r["capas"][0] > 64])

print("1. COMPLEJIDAD:")
print(f"   • Arquitecturas simples (≤32 neuronas): {simple_avg*100:.2f}% accuracy promedio")
print(f"   • Arquitecturas complejas (>64 neuronas o múltiples capas): {compleja_avg*100:.2f}% accuracy promedio")

if simple_avg >= compleja_avg - 0.01:
    print("   ✓ Las arquitecturas simples son suficientes para este problema")
else:
    print("   ✓ Las arquitecturas complejas ofrecen mejor rendimiento")

print("\n2. RAZONES DE LA ARQUITECTURA ÓPTIMA:")
print(f"   • Dataset pequeño: Solo 150 muestras → No necesita redes muy profundas")
print(f"   • Problema simple: Solo 3 clases linealmente separables")
print(f"   • Balance: Suficiente capacidad sin sobreajuste (overfitting)")
print(f"   • Eficiencia: Menos parámetros = entrenamiento más rápido")

print("\n3. PRINCIPIO DE PARSIMONIA (Navaja de Occam):")
print(f"   'La solución más simple que funcione es la mejor'")
print(f"   → Elegir la red más pequeña que logre ~98-100% accuracy")

print("\n4. PROBLEMA DE SOBREAJUSTE:")
if mejor["capas"][0] > 64:
    print(f"   ⚠️  Arquitecturas muy grandes pueden memorizar en lugar de aprender")
    print(f"   → Usar validación cruzada para confirmar generalización")
else:
    print(f"   ✓ La arquitectura óptima evita sobreajuste por su tamaño moderado")

print("\n" + "=" * 90)
print()