# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from scipy.signal import chirp
import matplotlib.pyplot as plt

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
PLOT_SIGNAL, PLOT_RESULT = False, True

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 5
BATCH_SIZE = 20
DROPOUT = 0
SEQ_LEN = 50  # Number of previous time steps to use as inputs in order to predict the output at the next time step
HIDDEN_SIZE = 16  # If you want to use a linear layer after the LSTM, give this some value. If not, give None
STATEFUL = True  # Both of these hyperparameters
LOSS_WHOLE_SEQ = True  # are explained below

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Generates a frequency-swept cosine signal (basically a cosine with changing frequency over time)
time_steps = np.linspace(0, 10, 5000)
x = chirp(time_steps, f0=1, f1=0.1, t1=10, method='linear')
if PLOT_SIGNAL:
    plt.plot(time_steps, x)
    plt.show()
# Splits into training and testing: we get the first 75% time steps as training and the rest as testing
x_train, x_test = x[:int(0.75*len(x))], x[int(0.75*len(x)):]
# Prepossesses the inputs on the required format (batch_size, timesteps, input_dim)
x_train_prep = np.empty((len(x_train)-SEQ_LEN, SEQ_LEN, 1))
if LOSS_WHOLE_SEQ:  # The targets are either the sequences shifted by one time step
    y_train_prep = np.empty((len(x_train)-SEQ_LEN, SEQ_LEN, 1))
else:  # or the single value of the signal at the time step that comes after the end of the input sequence
    y_train_prep = np.empty((len(x_train)-SEQ_LEN, 1))
for idx in range(len(x_train)-SEQ_LEN):
    x_train_prep[idx, :, :] = x_train[idx:SEQ_LEN+idx].reshape(-1, 1)
    if LOSS_WHOLE_SEQ:
        y_train_prep[idx, :] = x_train[idx+1:SEQ_LEN+idx+1].reshape(-1, 1)
    else:
        y_train_prep[idx, :] = x_train[SEQ_LEN+idx].reshape(1, 1)
x_test_prep = np.empty((len(x_test)-SEQ_LEN, SEQ_LEN, 1))
if LOSS_WHOLE_SEQ:
    y_test_prep = np.empty((len(x_test)-SEQ_LEN, SEQ_LEN, 1))
else:
    y_test_prep = np.empty((len(x_test) - SEQ_LEN, 1))
for idx in range(len(x_test)-SEQ_LEN):
    x_test_prep[idx, :, :] = x_test[idx:SEQ_LEN+idx].reshape(-1, 1)
    if LOSS_WHOLE_SEQ:
        y_test_prep[idx, :] = x_test[idx+1:SEQ_LEN+idx+1].reshape(-1, 1)
    else:
        y_test_prep[idx, :] = x_test[SEQ_LEN+idx].reshape(1, 1)
x_train, y_train, x_test, y_test = x_train_prep, y_train_prep, x_test_prep, y_test_prep
del x_train_prep, y_train_prep, x_test_prep, y_test_prep

# %% -------------------------------------- Training Prep ----------------------------------------------------------
if HIDDEN_SIZE is None:  # This layer takes an input of (batch_size, timesteps, input_dim).
    # In theory this should be enough (see PyTorch version) because the output function is a tanh and our signal is on
    # the interval [-1, 1]. Not sure why it's only learning half the signal...
    # Could it be due to some Keras internals regarding this function and the LSTM layer?
    model = Sequential([LSTM(units=1, dropout=DROPOUT, stateful=STATEFUL,
                             batch_input_shape=(BATCH_SIZE, SEQ_LEN, 1) if STATEFUL else None,
                             return_sequences=True if LOSS_WHOLE_SEQ else False)])
else:  # stateful=True means that it will take the hidden states of the previous batch as memory for the current batch
    # This is equivalent to passing the h_c_states as additional inputs in the PyTorch code.
    # Note that we need a constant batch_size in this case, but we cannot just drop the last batch as in PyTorch, so we
    # need to specify a batch_size such that both len(x_train) and len(x_test) are divisible by.
    model = Sequential([LSTM(units=HIDDEN_SIZE, dropout=DROPOUT, stateful=STATEFUL,
                             batch_input_shape=(BATCH_SIZE, SEQ_LEN, 1) if STATEFUL else None,
                             return_sequences=True if LOSS_WHOLE_SEQ else False),
                        Dense(1, activation="linear")])
    # return_sequences=True means that all the intermediate outputs of the LSTM will be returned, and thus the network
    # will be trained to take 1 previous time step as input and predict the next time step, but also to take 2
    # previous time steps as inputs and predict the next time step, and also 3, 4, ..., SEQ_LENGTH previous t_steps.
    # If False, only the last output will be returned. This means that the network will being trained specifically to
    # take SEQ_LENGTH previous time steps as inputs and predict the next time step.
model.compile(optimizer=Adam(lr=LR), loss="mean_squared_error")

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))

# %% ------------------------------------------ Final Test -------------------------------------------------------------
if PLOT_RESULT:
    # Gets all of the predictions one last time.
    pred_train = model.predict(x_train, batch_size=BATCH_SIZE)  # batch_size again in case STATEFUL=True
    pred_test = model.predict(x_test, batch_size=BATCH_SIZE)
    # Stores only the last prediction using each sequence [:SEQ_LEN, 1:SEQ_LEN+1, ...] as inputs. This is the default
    # behaviour if LOSS_WHOLE_SEQ=False
    if LOSS_WHOLE_SEQ:
        predictions_train, predictions_test = [], []
        for idx in range(len(pred_train)):
            predictions_train.append(pred_train[idx, -1].reshape(-1))
        for idx in range(len(pred_test)):
            predictions_test.append(pred_test[idx, -1].reshape(-1))
        pred_train, pred_test = np.array(predictions_train), np.array(predictions_test)
    # Plots the actual signal and the predicted signal using the previous SEQ_LEN points of the signal for each pred
    plt.plot(time_steps, x, label="Real Time Series", linewidth=2)
    plt.plot(time_steps[SEQ_LEN:len(pred_train)+SEQ_LEN],
             pred_train, linestyle='dashed', label="Train Prediction")
    plt.scatter(time_steps[len(pred_train)+2*SEQ_LEN:],
                pred_test, color="y", label="Test Prediction")
    plt.title("Chirp function with 1 Hz frequency at t=0 and 0.1 Hz freq at t=10\n"
              "LSTM predictions using the previous {} time steps".format(SEQ_LEN))
    plt.xlabel("Time"); plt.ylabel("Signal Intensity")
    plt.legend()
    plt.show()
