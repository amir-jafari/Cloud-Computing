# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import to_categorical


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
# Sets random seeds and some other stuff for reproducibility. Note even this might not give fully reproducible results.
# There seems to be a problem with the TF backend. However, the results should be very similar.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_NEURONS = (100, 200, 100)
N_EPOCHS = 20
BATCH_SIZE = 512
DROPOUT = 0.2

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Downloads from the internet and loads data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Reshapes to (n_examples, n_pixels), i.e, each pixel will be an input feature to the model
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Sequential([  # The dropout is placed right after the outputs of the hidden layers.
    Dense(N_NEURONS[0], input_dim=784, kernel_initializer=weight_init),  # This sets some of these outputs to 0, so that
    Activation("relu"),  # a random dropout % of the hidden neurons are not used during each training step,
    Dropout(DROPOUT),  # nor are they updated. The Batch Normalization normalizes the outputs from the hidden
    BatchNormalization()  # activation functions. This helps with neuron imbalance and can speed training significantly.
])  # Note this is an actual layer with some learnable parameters. It's not just min-maxing or standardizing.
# Loops over the hidden dims to add more layers
for n_neurons in N_NEURONS[1:]:
    model.add(Dense(n_neurons, activation="relu", kernel_initializer=weight_init))
    model.add(Dropout(DROPOUT, seed=SEED))
    model.add(BatchNormalization())
# Adds a final output layer with softmax to map to the 10 classes
model.add(Dense(10, activation="softmax", kernel_initializer=weight_init))
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
# Trains the MLP, while printing validation loss and metrics at each epoch
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
