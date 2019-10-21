# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from sklearn.metrics import accuracy_score


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
# Gets the device to move Tensors to (defaults to GPU (cuda) if available)
# If you want to use the CPU even when you have GPU, just do device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Sets random seeds and some other stuff for reproducibility
torch.manual_seed(42)  # Note that this does not always ensure reproducible results
np.random.seed(42)  # (See https://pytorch.org/docs/stable/notes/randomness.html)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_NEURONS = (100, 200, 100)
N_EPOCHS = 20
BATCH_SIZE = 512
DROPOUT = 0.2


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def acc(x, y, return_labels=False):
    """ Simple function to get the accuracy or the predicted labels. The label with the highest logit is chosen """
    with torch.no_grad():  # Explained on the training loop
        logits = model(x)  # (n_examples, n_labels) --> Need to operate on columns, so axis=1
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
class MLP(nn.Module):
    """ MLP with 3 hidden layers """
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3, dropout=DROPOUT):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784, hidden_dim1)
        # The Batch Normalization normalizes the outputs from the hidden activation functions. This helps with neuron
        self.linear1_bn = nn.BatchNorm1d(hidden_dim1)  # imbalance and can speed training significantly. Note this is
        # an actual layer with some learnable parameters. It's not just min-maxing or standardizing
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear2_bn = nn.BatchNorm1d(hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.linear3_bn = nn.BatchNorm1d(hidden_dim3)
        self.out = nn.Linear(hidden_dim3, 10)
        self.act = torch.relu
        # The dropout is placed right after the outputs of the hidden layers. This sets some of these
        self.drop = nn.Dropout(dropout)  # outputs to 0, so that a random dropout % of the hidden
        # neurons are not used during each training step, nor are they updated

    def forward(self, x):
        out = self.drop(self.linear1_bn(self.act(self.linear1(x))))
        out = self.drop(self.linear2_bn(self.act(self.linear2(out))))
        return self.out(self.drop(self.linear3_bn(self.act(self.linear3(out)))))


# Re-implements the MLP class with an arbitrary number of hidden layers. This way we don't need
# to create another class or modify the existing one if we want to try out more or less layers
class MLPModuleList(nn.Module):
    """ MLP with len(neurons_per_layer) hidden layers """
    def __init__(self, neurons_per_layer, dropout=DROPOUT):
        super(MLPModuleList, self).__init__()
        dims = (784, *neurons_per_layer)  # Gets a tuple of (784, neurons_layer1, neurons_layer2, ...)
        self.layers = nn.ModuleList([  # Uses ModuleList to get a list of an arbitrary number of layers based on
            nn.Sequential(  # the length of neurons_per_layer. A simple python list does not work because
                nn.Linear(dims[i], dims[i+1]),  # model.parameters() will not look inside the list.
                nn.ReLU(),  # Sequential adds the operations we want one after the other under the same forward pass
                nn.BatchNorm1d(dims[i+1]),
                nn.Dropout(dropout)
            ) for i in range(len(dims)-1)
        ])
        self.layers.extend(nn.ModuleList([nn.Linear(neurons_per_layer[-1], 10)]))  # Equivalent to self.out on MLP class

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Downloads from the internet and loads data
# Reshapes to (n_examples, n_pixels), i.e, each pixel will be an input feature to the model
# Moves to device
data_train = datasets.MNIST(root='.', train=True, download=True)
x_train, y_train = data_train.data.view(len(data_train), -1).float().to(device), data_train.targets.to(device)
x_train.requires_grad = True
data_test = datasets.MNIST(root='.', train=False, download=True)
x_test, y_test = data_test.data.view(len(data_test), -1).float().to(device), data_test.targets.to(device)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
# Using MLP instead of MLPModuleList will give exactly the same results, as we have the exact same architecture
model = MLP(*N_NEURONS).to(device)
# model = MLPModuleList(N_NEURONS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):

    loss_train = 0  # Initializes train loss which will be added up after going forward on each batch
    model.train()  # Activates Dropout and makes BatchNorm use the actual training data to compute the mean and std
    # (this is the default behaviour but will be changed later on the evaluation phase)
    for batch in range(len(x_train)//BATCH_SIZE + 1):  # Loops over the number of batches (n_examples//batch_size)
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)  # Gets a slice to index the data
        optimizer.zero_grad()
        logits = model(x_train[inds])
        loss = criterion(logits, y_train[inds])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    model.eval()  # Deactivates Dropout and makes BatchNorm use mean and std estimates computed during training
    with torch.no_grad():  # The code inside will run without Autograd, which reduces memory usage, speeds up
        y_test_pred = model(x_test)  # computations and makes sure the model can't use the test data to learn
        loss = criterion(y_test_pred, y_test)
        loss_test = loss.item()

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, loss_train/batch, acc(x_train, y_train), loss_test, acc(x_test, y_test)))
