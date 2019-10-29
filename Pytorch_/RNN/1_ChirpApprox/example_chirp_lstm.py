# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import chirp
import matplotlib.pyplot as plt

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
PLOT_SIGNAL, PLOT_RESULT = False, True

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-2
N_EPOCHS = 100
BATCH_SIZE = 128
DROPOUT = 0
SEQ_LEN = 50  # Number of previous time steps to use as inputs in order to predict the output at the next time step
HIDDEN_SIZE = 1  # If you want to use a linear layer after the LSTM, you can use a hidden size greater than 1
N_LAYERS = 1
STATEFUL = False  # Both of these hyperparameters
LOSS_WHOLE_SEQ = False  # are explained below


# %% -------------------------------------- LSTM Class -----------------------------------------------------------------
class ChirpLSTM(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS, dropout=DROPOUT):
        super(ChirpLSTM, self).__init__()
        # The inputs to this layer are:
        # 1. The current input sequence, of shape (seq_len, batch, input_size)
        # 2. The last hidden state/s of the layer/s on the previous time step, of shape (n_layer, batch, hidden_size)
        # 3. The last cell state/s of the layer/s on the previous time step, of shape (n_layer, batch, hidden_size)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)
        # The outputs are:
        # 1. The hidden states of the last layer on the current time step, of shape (seq_len, batch, hidden_size)
        # 2. The last hidden state/s of the layer/s on the current time step, of shape (n_layer, batch, hidden_size)
        # 3. The last cell state/s of the layer/s on the current time step, of shape (n_layer, batch, hidden_size)
        # NOTE: Each hidden state is literally the output of the network at each time step
        # self.out = nn.Linear(hidden_size, 1)  # This linear layer would be needed if the signal we wanted to predict
        # was not on the range [-1, 1], as the output function of a LSTM is a tanh.

    def forward(self, p, hidden_state, cell_state):
        lstm_out, h_c_states = self.lstm(p, (hidden_state, cell_state))
        return lstm_out, h_c_states
        # return self.out(lstm_out), h_c_states


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Generates a frequency-swept cosine signal (basically a cosine with changing frequency over time)
time_steps = np.linspace(0, 10, 5000)
x = chirp(time_steps, f0=1, f1=0.1, t1=10, method='linear')
if PLOT_SIGNAL:
    plt.plot(time_steps, x)
    plt.show()
# Splits into training and testing: we get the first 75% time steps as training and the rest as testing
x_train, x_test = x[:int(0.75*len(x))], x[int(0.75*len(x)):]
# Prepossesses the inputs on the required format (seq_len, batch, input_size)
x_train_prep = np.empty((SEQ_LEN, len(x_train)-SEQ_LEN, 1))
for idx in range(len(x_train)-SEQ_LEN):
    x_train_prep[:, idx, :] = x_train[idx:SEQ_LEN+idx].reshape(-1, 1)
x_test_prep = np.empty((SEQ_LEN, len(x_test)-SEQ_LEN, 1))
for idx in range(len(x_test)-SEQ_LEN):
    x_test_prep[:, idx, :] = x_test[idx:SEQ_LEN+idx].reshape(-1, 1)
x_train, x_test = torch.Tensor(x_train_prep).to(device), torch.Tensor(x_test_prep).to(device)
x_train.requires_grad = True
del x_train_prep, x_test_prep

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = ChirpLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):

    h_state = torch.zeros(N_LAYERS, BATCH_SIZE, HIDDEN_SIZE).float().to(device)  # Initializes the hidden
    c_state = torch.zeros(N_LAYERS, BATCH_SIZE, HIDDEN_SIZE).float().to(device)  # and cell states
    loss_train = 0
    model.train()
    # Here we don't do x_train.shape[1]//BATCH_SIZE + 1 because each batch must have the same dim, due to the hidden
    for batch in range(x_train.shape[1]//BATCH_SIZE):  # and cell states. We could alternately pass a smaller h_c_state
        # to the last batch, initialized to zeros, same as in lines 80 and 81.
        inp_inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)  # The sequence starting on the next time step is
        tar_inds = slice(batch*BATCH_SIZE+1, (batch+1)*BATCH_SIZE+1)  # our target sequence
        optimizer.zero_grad()
        pred, h_c_state = model(x_train[:, inp_inds, :], h_state, c_state)
        if STATEFUL:
            # Detaches the last hidden and cell states from the graph before passing them to the next forward pass
            h_state, c_state = h_c_state[0].detach(), h_c_state[1].detach()
        else:  # STATEFUL means that it will take the hidden states of the previous batch
            pass  # as memory for the current batch. If not, we just pass the random states.
        if LOSS_WHOLE_SEQ:  # Computes the loss with all the hidden states and their corresponding targets, and not only
            loss = criterion(pred, x_train[:, tar_inds, :])  # the last hidden state. This means that the network is
        # being trained to take 1 previous time step as input and predict the next time step, but also to take 2
        # previous time steps as inputs and predict the next time step, and also 3, 4, ..., SEQ_LENGTH previous t_steps
        else:  # Uses only the last hidden state and its corresponding target to compute the loss. This means that the
            loss = criterion(pred[-1], x_train[-1, tar_inds, :])  # network is being trained specifically to take
        # SEQ_LENGTH previous time steps as input and predict the next time step
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    h_state = torch.zeros(N_LAYERS, x_test.shape[1]-1, HIDDEN_SIZE).float().to(device)
    c_state = torch.zeros(N_LAYERS, x_test.shape[1]-1, HIDDEN_SIZE).float().to(device)
    model.eval()
    with torch.no_grad():
        pred, h_c_state = model(x_test[:, :-1, :], h_state, c_state)
        loss = criterion(pred, x_test[:, 1:, :])
        loss_test = loss.item()

    print("Epoch {} | Train Loss {:.5f} - Test Loss {:.5f}".format(epoch, loss_train/batch, loss_test))

# %% ------------------------------------------ Final Test -------------------------------------------------------------
if PLOT_RESULT:
    # Gets all of the predictions one last time
    with torch.no_grad():
        h_state = torch.zeros(N_LAYERS, x_train.shape[1], HIDDEN_SIZE).float().to(device)
        c_state = torch.zeros(N_LAYERS, x_train.shape[1], HIDDEN_SIZE).float().to(device)
        pred_train, _ = model(x_train, h_state, c_state)
        h_state = torch.zeros(N_LAYERS, x_test.shape[1], HIDDEN_SIZE).float().to(device)
        c_state = torch.zeros(N_LAYERS, x_test.shape[1], HIDDEN_SIZE).float().to(device)
        pred_test, _ = model(x_test, h_state, c_state)
    # Stores only the last prediction using each sequence [:SEQ_LEN, 1:SEQ_LEN+1, ...] as inputs
    predictions_train, predictions_test = [], []
    for idx in range(pred_train.shape[1]):
        predictions_train.append(pred_train[-1, idx].reshape(-1))
    for idx in range(pred_test.shape[1]):
        predictions_test.append(pred_test[-1, idx].reshape(-1))
    # Plots the actual signal and the predicted signal using the previous SEQ_LEN points of the signal for each pred
    plt.plot(time_steps, x, label="Real Time Series", linewidth=2)
    plt.plot(time_steps[SEQ_LEN:len(predictions_train)+SEQ_LEN],
             np.array(predictions_train), linestyle='dashed', label="Train Prediction")
    plt.scatter(time_steps[len(predictions_train)+2*SEQ_LEN:],
                np.array(predictions_test), color="y", label="Test Prediction")
    plt.title("Chirp function with 1 Hz frequency at t=0 and 0.1 Hz freq at t=10\n"
              "LSTM predictions using the previous {} time steps".format(SEQ_LEN))
    plt.xlabel("Time"); plt.ylabel("Signal Intensity")
    plt.legend()
    plt.show()
