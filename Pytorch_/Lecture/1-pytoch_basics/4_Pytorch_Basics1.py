import torch
import torch.nn as nn
from torch.autograd import Variable

#======================= Basic autograd example 1 =======================#
# Create tensors.
p = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

# Build a computational graph.
y = w * p + b

# Compute gradients.
y.backward()

# Print out the gradients.
print(p.grad)
print(w.grad)
print(b.grad)


#======================== Basic autograd example 2 =======================#
# Create tensors.
p = Variable(torch.randn(5, 3))
y = Variable(torch.randn(5, 2))

# Build a linear layer.
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# Build Loss and Optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward propagation.
pred = linear(p)

# Compute loss.
performance_index = criterion(pred, y)
print('loss: ', performance_index.item())

# Backpropagation.
performance_index.backward()

# Print out the gradients.
print ('de/dw: ', linear.weight.grad)
print ('de/db: ', linear.bias.grad)

# 1-step Optimization (gradient descent).
optimizer.step()


# Print out the loss after optimization.
pred = linear(p)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())