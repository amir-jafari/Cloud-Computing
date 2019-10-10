import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
rng = np.random

input = np.array([[1],[2],[3],[4]])
target = np.array([[3],[5],[7],[9]])

p = T.dmatrix("p")
t = T.dmatrix("t")

# initialize the weight
w = theano.shared(rng.randn(1), name="w")

# initialize the bias
b = theano.shared(0., name="b")

iterations = 1000
totperf = np.zeros((iterations,))

a = w*p + b

e = t - a

e2 = T.sqr(e)

perf = T.sum(e2)

gw, gb = T.grad(perf, [w, b])

train = theano.function(
          inputs=[p,t],
          outputs=[a, perf],
          updates=((w, w - 0.01 * gw), (b, b - 0.01 * gb)))
predict = theano.function(inputs=[p], outputs=a)
perform = theano.function(inputs=[p,t], outputs=perf)
perftest = perform(input,target)
print(perftest)

# Train,
for i in range(iterations):
    ii = i%4
    pred, err = train(input, target)
    totperf[i] = err

print("Final model:")
print(w.get_value())
print(b.get_value())
plt.figure(1)
plt.semilogy(np.arange(iterations), totperf)
plt.xlabel('Number of Iteration')
plt.ylabel('Training Loss Values')
plt.title('Training Loss')
plt.show()

atest = predict([[2.2],[3.5]])
print(atest)