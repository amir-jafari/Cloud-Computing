# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Add
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

# %% -------------------------------------- Training Prep ----------------------------------------------------------
# Gets an instance of Input class with the right shape
inputs = Input(shape=(784,))
# Defines a succession of layers
x = Dense(N_NEURONS[0], activation='relu')(inputs)
x = Dense(784, activation='relu')(x)
x = Dropout(DROPOUT)(x)
# We add the original input to the output of the dropout before BatchNormalization
x = Add()([x, inputs])
x = BatchNormalization()(x)
x = Dense(N_NEURONS[1], activation='relu')(x)
x = Dropout(DROPOUT)(x)
probs = Dense(10, activation="softmax")(x)
# Gets an instance of Model class based on inputs and probs, which has kept track of all the operations in-between
model = Model(inputs=inputs, outputs=probs)
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
