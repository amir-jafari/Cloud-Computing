# # %%---------------------------------------------------------------------------------------------
import keras
from keras import layers
import numpy as np
x = np.linspace(-4,4,500)
y = np.sin(x)

inputs = keras.Input(shape=(1,))
x1 =layers.Dense(10,activation='relu')(inputs)
ouputs = layers.Dense(1, activation='relu')(x1)

model = keras.Model(inputs, ouputs)
model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])
model.fit(x, y, epochs=10)
