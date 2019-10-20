# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.02
N_NEURONS = 2
N_EPOCHS = 10000
PRINT_LOSS_EVERY = 1000


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2, hidden_dim)
        self.act1 = torch.sigmoid
        self.linear2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.linear2(self.act1(self.linear1(x)))


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
p = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
t = np.array([0, 0, 1, 1])
p = torch.FloatTensor(p)
p.requires_grad = True
t = torch.Tensor(t).long()

# %% -------------------------------------- Training Prep --------------------------------------------------------------
model = MLP(N_NEURONS)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# A classification problem is usually approached with a Categorical Cross-Entropy performance index
criterion = nn.CrossEntropyLoss()  # This one combines both the loss and the Log-SoftMax output function,
# which computes the probabilities of each example belonging to each class
# This is the preferred way as its computation is more stable. Thus, there is no need to include the
# SoftMax/Log-SoftMax output function on the model itself

# %% -------------------------------------- Training Loop --------------------------------------------------------------
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    logits = model(p)
    loss = criterion(logits, t)
    loss.backward()
    optimizer.step()
    if epoch % PRINT_LOSS_EVERY == 0:
        print("Epoch {} | Loss {:.5f}".format(epoch, loss.item()))

# %% -------------------------------------- Check Approx ---------------------------------------------------------------
print(accuracy_score(t.numpy(), np.argmax(logits.detach().numpy(), axis=1))*100)

# Gets weights and biases, and from them the DBs, and plots them
w, b = model.linear1.weight.detach().numpy(), model.linear1.bias.detach().numpy()
a = np.arange(0, 1.1, 0.1)
db1 = -w[0, 0]*a/w[0, 1] - b[0]/w[0, 1]
db2 = -w[1, 0]*a/w[1, 1] - b[1]/w[1, 1]
p_np = p.detach().cpu().numpy()
plt.scatter(p_np[:2, 0], p_np[:2, 1], marker="*", c="b")
plt.scatter(p_np[2:, 0], p_np[2:, 1], marker=".", c="b")
plt.plot(a, db1, c="b")
plt.plot(a, db2, c="b")
plt.show()
