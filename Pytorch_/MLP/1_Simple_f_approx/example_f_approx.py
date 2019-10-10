# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
# It is always good practice to set the hyper-parameters at the beginning of the script
# And even better to define a params class if the script is long and complex
LR = 2.5e-1
N_NEURONS = 10
N_EPOCHS = 50000
PRINT_LOSS_EVERY = 1000


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
# Defines model class. Inherits from nn.Module to get all useful methods and make the class compatible with PyTorch
class MLP(nn.Module):
    """ MLP with 2 layers, sigmoid transfer function and hidden_dim neurons on the hidden layer"""
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()  # Initializes nn.Module
        # Linear Layer that maps input (n_examples, 1) to hidden_dim (number of neurons)
        self.linear1 = nn.Linear(1, hidden_dim)
        self.act1 = torch.sigmoid  # Hidden transfer function
        # Linear Layer that maps hidden_dim to output (n_examples, 1), i.e, 1 neuron
        self.linear2 = nn.Linear(hidden_dim, 1)

    # Calling this method "forward" will make it possible to go forward by just calling model(x) (see below)
    def forward(self, x):
        # Just a sequential pass to go through the 2 layers
        return self.linear2(self.act1(self.linear1(x)))


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Gets the input interval and the targets
p = np.linspace(-2, 2, 100)  # (100,)
t = np.exp(p) - np.sin(2*np.pi*p)  # (100,)
# Converts to Tensors and reshapes to suitable shape (n_examples, 1)
# requires_grad=True on the input so that the gradients are computed when calling loss.backward()
# i.e, so that all the operations performed on p and on their outputs are made part of the Computational Graph
p = torch.Tensor(p).reshape(-1, 1)
p.requires_grad = True
t = torch.Tensor(t).reshape(-1, 1)

# %% -------------------------------------- Training Prep --------------------------------------------------------------
# Initializes model and moves it to GPU if available
model = MLP(N_NEURONS)
# Initializes a Gradient Descent Optimizer with default hyper-parameters
# We pass the model.parameters() generator so that all the model's parameters are updated
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# Initializes a Mean Square Error performance index
criterion = nn.MSELoss()

# %% -------------------------------------- Training Loop --------------------------------------------------------------
# Starts the training loop
for epoch in range(N_EPOCHS):
    # Sets the gradients stored on the .grad attribute of each parameter from the previous iteration to 0
    optimizer.zero_grad()  # It is good practice to do it right before going forward on any model
    # Goes forward (doing full batch here), notice we don't need to do model.forward(p)
    t_pred = model(p)
    # Computes the mse
    loss = criterion(t, t_pred)
    # Goes backwards (computes all the gradients of the mse w.r.t the parameters
    # starting from the output layer all the way to the input layer)
    loss.backward()
    # Updates all the parameters using the gradients which were just computed
    optimizer.step()
    # Checks the training process
    if epoch % PRINT_LOSS_EVERY == 0:
        print("Epoch {} | Loss {:.5f}".format(epoch, loss.item()))

# %% -------------------------------------- Check Approx ---------------------------------------------------------------
# Visually shows the approximation obtained by the MLP
plt.title("MLP fit to $y = e^x - sin(2 \\pi x)$ | MSE: {:.5f}".format(loss.item()))
plt.xlabel("x")
plt.ylabel("y")
# .detach() to take the Tensors out of the computational graph
# .numpy() to convert the Tensors to NumPy arrays
plt.plot(p.detach().numpy(), t.numpy(), label="Real Function")
plt.plot(p.detach().numpy(), t_pred.detach().numpy(), linestyle="dashed", label="MLP Approximation")
plt.legend()
plt.show()
