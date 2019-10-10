import keras
from keras import layers
import numpy as np
x = np.linspace(-4,4,500)
y = np.sin(x)

model = keras.Sequential()

model.add(layers.Dense(10, activation='relu' , input_shape=(1,)))
model.add(layers.Dense(1, activation='linear' , input_shape=(10,)))

model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])
model.fit(x, y, epochs=10)
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
# %%---------------------------------------------------------------------------------------------
import keras
from keras import layers
import numpy as np
x = np.linspace(-4,4,500).reshape(-1,1)
y = np.sin(x)

class Mymodel(keras.Model):
    def __init__(self):
        super(Mymodel,self).__init__()
        self.dense1 = layers.Dense(1, activation='relu')
        self.dense2 = layers.Dense(10, activation='relu')
        self.dense3 = layers.Dense(1, activation='linear')
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)
        return output

model = Mymodel()

model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])
model.fit(x, y, epochs=10)
# %%---------------------------------------------------------------------------------------------
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])