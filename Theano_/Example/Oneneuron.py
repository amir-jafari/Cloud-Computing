import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

#Define variables:
x = T.vector('x')
w = T.vector('w')
b = T.scalar('b')

#Define mathematical expression:
z = T.dot(x,w)+b
a = ifelse(T.lt(z,0),0,1)

neuron = theano.function([x,w,b],a)
#Define inputs and weights
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
weights = [ 1, 1]
bias = -1.5

#Iterate through all inputs and find outputs:
for i in range(len(inputs)):
    t = inputs[i]
    out = neuron(t,weights,bias)
    print 'The output for x1=%d | x2=%d is %d' % (t[0],t[1],out)
