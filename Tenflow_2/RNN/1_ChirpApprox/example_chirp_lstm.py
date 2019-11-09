# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from scipy.signal import chirp
import matplotlib.pyplot as plt

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)
PLOT_SIGNAL, PLOT_RESULT = False, True


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 5
BATCH_SIZE = 32
DROPOUT = 0
SEQ_LEN = 50  # Number of previous time steps to use as inputs in order to predict the output at the next time step
HIDDEN_SIZE = 16  # If you want to use a linear layer after the LSTM, give this some value. If not, give None
STATEFUL = False  # Both of these hyperparameters
LOSS_WHOLE_SEQ = False  # are explained below

# %% -------------------------------------- LSTM Class -----------------------------------------------------------------
class ChirpLSTM(tf.keras.Model):
    def __init__(self, dropout=DROPOUT):
        super(ChirpLSTM, self).__init__()
        # This layer takes an input of (batch_size, timesteps, input_dim).
        self.lstm = tf.keras.layers.LSTM(units=HIDDEN_SIZE if HIDDEN_SIZE is not None else 1, dropout=dropout,
                                         batch_input_shape=(BATCH_SIZE, SEQ_LEN, 1) if STATEFUL else (None, SEQ_LEN, 1),
                                         return_sequences=True if LOSS_WHOLE_SEQ else False, stateful=STATEFUL)
        # stateful=True means that it will take the hidden states of the previous batch as memory for the current batch
        # This is equivalent to passing the h_c_states as additional inputs in the PyTorch code.
        # return_sequences=True means that all the intermediate outputs of the LSTM will be returned, and thus the network
        # will be trained to take 1 previous time step as input and predict the next time step, but also to take 2
        # previous time steps as inputs and predict the next time step, and also 3, 4, ..., SEQ_LENGTH previous t_steps.
        # If False, only the last output will be returned. This means that the network will being trained specifically to
        # take SEQ_LENGTH previous time steps as inputs and predict the next time step.
        self.out = tf.keras.layers.Dense(1, activation="linear")
        self.training = True

    def call(self, x):
        lstm_out = self.lstm(x, training=self.training)
        if HIDDEN_SIZE:
            return self.out(lstm_out)
        else:  # In theory this should be enough (see PyTorch version) because the output function is a tanh and our
            # signal is on the interval [-1, 1]. Not sure why it's only learning half the signal...
            # Could it be due to some Keras internals regarding this function and the LSTM layer?
            return lstm_out

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
for idx in range(len(x_train)-SEQ_LEN):
    x_train_prep[idx, :, :] = x_train[idx:SEQ_LEN+idx].reshape(-1, 1)
x_test_prep = np.empty((len(x_test)-SEQ_LEN, SEQ_LEN, 1))
for idx in range(len(x_test)-SEQ_LEN):
    x_test_prep[idx, :, :] = x_test[idx:SEQ_LEN+idx].reshape(-1, 1)
x_train, x_test = tf.cast(tf.convert_to_tensor(x_train_prep), tf.float32), tf.cast(tf.convert_to_tensor(x_test_prep), tf.float32)
del x_train_prep, x_test_prep

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = ChirpLSTM()
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
criterion = tf.keras.losses.MeanSquaredError()

train_loss = tf.keras.metrics.Mean(name='train_loss')
@tf.function
def train(x, y, loss_whole_seq=LOSS_WHOLE_SEQ):
    model.training = True
    with tf.GradientTape() as tape:
        pred = model(x)
        if loss_whole_seq:
            loss = criterion(y, pred)
        else:
            loss = criterion(y[:, -1], pred[:, -1])
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

test_loss = tf.keras.metrics.Mean(name='test_loss')
@tf.function
def eval(x, y, loss_whole_seq=LOSS_WHOLE_SEQ):
    model.training = False
    pred = model(x)
    if loss_whole_seq:
        loss = criterion(y, pred)
    else:
        loss = criterion(y[:, -1], pred[:, -1])
    test_loss(loss)

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):

    if STATEFUL:
        pass
        # This should also be possible and should be done, but there is some weird error
        # model.lstm.reset_states([tf.zeros((BATCH_SIZE, HIDDEN_SIZE)), tf.zeros((BATCH_SIZE, HIDDEN_SIZE))])
        # model.lstm.states = [tf.zeros((BATCH_SIZE, HIDDEN_SIZE)), tf.zeros((BATCH_SIZE, HIDDEN_SIZE))] does not work either
    for batch in range(len(x_train) // BATCH_SIZE):
        inp_inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        # The targets are either the sequences shifted by one time step
        tar_inds = slice(batch * BATCH_SIZE + 1, (batch + 1) * BATCH_SIZE + 1)
        # or the single value of the signal at the time step that comes after the end of the input sequence
        train(x_train[inp_inds], x_train[tar_inds])

    if STATEFUL:
        # This should be possible but apparently it is not...
        # model.lstm.reset_states([tf.zeros((len(x_test)-1, HIDDEN_SIZE)), tf.zeros((len(x_test)-1, HIDDEN_SIZE))])
        # eval(x_test[:-1], x_test[1:])

        # This should also be possible and should be done, but again apparently is not...
        # model.lstm.reset_states([tf.zeros((BATCH_SIZE, HIDDEN_SIZE)), tf.zeros((BATCH_SIZE, HIDDEN_SIZE))])
        for batch in range(len(x_test) // BATCH_SIZE):
            inp_inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
            tar_inds = slice(batch * BATCH_SIZE + 1, (batch + 1) * BATCH_SIZE + 1)
            eval(x_test[inp_inds], x_test[tar_inds])
    else:
        eval(x_test[:-1], x_test[1:])

    print("Epoch {} | Train Loss {:.5f} - Test Loss {:.5f}".format(epoch, train_loss.result(), test_loss.result()))
    train_loss.reset_states(); test_loss.reset_states()

# %% ------------------------------------------ Final Test -------------------------------------------------------------
if PLOT_RESULT:
    model.training = False
    # Gets all of the predictions one last time.
    if STATEFUL:  # This ugly conditional could be avoided if the stateful stuff worked well on TF2... plus we are
        if LOSS_WHOLE_SEQ:  # missing the last batch and even using hidden states from training into testing...
            pred_train = np.empty((len(x_train)//BATCH_SIZE*BATCH_SIZE, SEQ_LEN, 1))  # See PyTorch or Keras versions
            pred_test = np.empty((len(x_test) // BATCH_SIZE * BATCH_SIZE, SEQ_LEN, 1))  # for ideal way of doing this
        else:
            pred_train = np.empty((len(x_train) // BATCH_SIZE * BATCH_SIZE, 1))
            pred_test = np.empty((len(x_test) // BATCH_SIZE * BATCH_SIZE, 1))
        for batch in range(len(x_train) // BATCH_SIZE):
            inp_inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
            if LOSS_WHOLE_SEQ:
                pred_train[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE, :, :] = model(x_train[inp_inds])
            else:
                pred_train[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE, :] = model(x_train[inp_inds])
        for batch in range(len(x_test) // BATCH_SIZE):
            inp_inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
            if LOSS_WHOLE_SEQ:
                pred_test[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE, :, :] = model(x_test[inp_inds])
            else:
                pred_test[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE, :] = model(x_test[inp_inds])
    else:
        pred_train = model(x_train).numpy()
        pred_test = model(x_test).numpy()
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
    if STATEFUL:
        plt.scatter(time_steps[len(pred_train) + 2 * SEQ_LEN + BATCH_SIZE:-(
                len(x) - (len(pred_train) + 2 * SEQ_LEN + BATCH_SIZE) - len(pred_test))],
                    pred_test, color="y", label="Test Prediction")
    else:
        plt.scatter(time_steps[len(pred_train)+2*SEQ_LEN:],
                    pred_test, color="y", label="Test Prediction")
    plt.title("Chirp function with 1 Hz frequency at t=0 and 0.1 Hz freq at t=10\n"
              "LSTM predictions using the previous {} time steps".format(SEQ_LEN))
    plt.xlabel("Time"); plt.ylabel("Signal Intensity")
    plt.legend()
    plt.show()
