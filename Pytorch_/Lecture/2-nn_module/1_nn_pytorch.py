#----------------------------------------------------------------------------
import torch
#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a_size = 10              # Network output size
#----------------------------------------------------------------------------
p = torch.randn(Batch_size, R)
t = torch.randn(Batch_size, a_size, requires_grad=False)
#----------------------------------------------------------------------------
model = torch.nn.Sequential(
    torch.nn.Linear(R, S),
    torch.nn.ReLU(),
    torch.nn.Linear(S, a_size),
)
#----------------------------------------------------------------------------
performance_index = torch.nn.MSELoss(reduction='sum')
#----------------------------------------------------------------------------
learning_rate = 1e-4
for index in range(500):

    a = model(p)
    loss = performance_index(a, t)
    print(index, loss.item())

    model.zero_grad()
    loss.backward()

    for param in model.parameters():
        param.data -= learning_rate * param.grad