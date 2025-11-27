import numpy as np
#from sklearn import preprocessing

X = np.array([[0,0],[0,1],[1,0],[1,1]],"float32") 
Y = np.array([0,1,1,0], "float32")

# Definir Modelo
from tensorflow import random
from tensorflow.keras import Input, Model, Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
np.random.seed(32)

model = Sequential()
model.add(Dense(80, activation='relu', input_dim=2))
model.add(Dense(100, activation='relu'))
#model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compila el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ajuste del modelo
model.fit(X, Y, epochs=350) #, batch_size=4)

# Se evalua el modelo
loss, accuracy = model.evaluate(X, Y, verbose=0)
print(loss)
print(accuracy)

y_predicted = model.predict(X)
print(y_predicted)