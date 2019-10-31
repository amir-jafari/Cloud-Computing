## Simple Example: MLP as function approximator

`example_f_approx.py` trains a Multilayer Perceptron to approximate the function $y = e^x - \sin(2 \pi x)$ on the interval [-2, 2]. It does so by generating 100 targets using this function on 100 equally spaced points on such interval, and then fitting the MLP to this data using Batch Gradient Descent via backpropagation to minimize a Mean Square Error performance index. The MLP contains 2 layers, a sigmoid hidden transfer function, 10 hidden neurons and 1 output neuron with an identity output function. That is, $S^1 = 10$, $S^2 = 1$ and $R=1$. The last two are set by the problem.

### New Code

- Most basic hyper-parameters.
- Creating own model class by subclassing `tf.keras.Model`. Includes linear layers and transfer functions.
- Preparing data in Tensor format.
- Basic training loop, using optimizer and loss function. `with tf.GradientTape()` context manager.

## Simple Exercise: MLP as 3D function approximator

`exercise_f_approx3d.py` asks to play around with the MLP class and also takes the example one step further to fit a function of 2 variables: $y = x_1^2 - x_2^2$. A solution can be found in `exercise_solution_f_approx3d.py`

### New Code

- Helper functions
- 3D plotting