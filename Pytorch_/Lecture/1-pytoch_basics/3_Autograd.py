import torch
from torch.autograd import Variable
#----------------------------------------------------------------------------
dtype = torch.float
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a = 10              # Network output size
#----------------------------------------------------------------------------
p = torch.randn(Batch_size, R, device=device, dtype=dtype, requires_grad=False)
t = torch.randn(Batch_size, a, device=device, dtype=dtype, requires_grad=False)
#----------------------------------------------------------------------------
w1 = torch.randn(R, S, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(S, a, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
#----------------------------------------------------------------------------
for index in range(500):

    a_net = p.mm(w1).clamp(min=0).mm(w2)
    loss = (a_net - t).pow(2).sum()

    print(index, loss.item())
    loss.backward()

    with torch.no_grad():

        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()