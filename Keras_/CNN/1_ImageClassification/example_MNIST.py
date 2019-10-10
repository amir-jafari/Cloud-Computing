# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
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
N_EPOCHS = 30
BATCH_SIZE = 512
DROPOUT = 0.5

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Reshapes to (n_examples, n_channels, height_pixels, width_pixels)
x_train, x_test = x_train.reshape(len(x_train), 1, 28, 28), x_test.reshape(len(x_test), 1, 28, 28)
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Sequential([
    Conv2D(16, 3, data_format="channels_first", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2),
    Conv2D(32, 3, data_format="channels_first", activation="relu"),
    BatchNormalization(),
    AveragePooling2D(2),
    Flatten(),
    Dense(400, activation="tanh"),
    Dropout(DROPOUT),
    BatchNormalization(),
    Dense(10, activation="softmax")
])
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
