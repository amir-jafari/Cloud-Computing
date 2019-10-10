# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import to_categorical


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_NEURONS = (100, 100)
N_EPOCHS = 20
BATCH_SIZE = 512
DROPOUT = 0.2

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)


# %% -------------------------------------- Training Prep --------------------------------------------------------------
class MLP(Model):
    # This model is equivalent to the one defined on 4_FuncionalAPI/example_MNIST.py
    # Notice we don't need to use the Input class nor the Add layer
    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = Dense(N_NEURONS[0], activation='relu')
        self.dense2 = Dense(784, activation='relu')
        self.drop = Dropout(DROPOUT)
        self.bn = BatchNormalization()
        self.dense3 = Dense(N_NEURONS[1], activation="relu")
        self.out = Dense(10, activation="softmax")

    def call(self, x):  # This method will be called inside model.fit()
        xx = self.drop(self.dense2(self.dense1(x)))
        xx = self.bn(xx + x)  # Replaces the Add layer just with + operation
        return self.out(self.drop(self.dense3(xx)))


model = MLP()
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
