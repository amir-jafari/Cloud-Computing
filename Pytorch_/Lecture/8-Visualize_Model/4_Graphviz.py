#----------------------------------------------------------------------------
import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace
import webbrowser
#----------------------------------------------------------------------------
#https://github.com/szagoruyko/pytorchviz/blob/master/examples.ipynb
#----------------------------------------------------------------------------
# sudo pip3 install graphviz
# sudo pip3 install git+https://github.com/szagoruyko/pytorchviz
#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a_size = 10              # Network output size
#----------------------------------------------------------------------------
model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1,8)

a =make_dot(model(x), params=dict(model.named_parameters()))

print(a.source)
b =a.render('graph')
webbrowser.open_new(r'/home/ajafari/Deep-Learning/graph.pdf')

with torch.onnx.set_training(model, False):
    trace, _ = torch.jit.get_trace_graph(model, args=(x,))
c = make_dot_from_trace(trace)
print(c.source)
d =c.render('graph1')
webbrowser.open_new(r'/home/ajafari/Deep-Learning/graph1.pdf')