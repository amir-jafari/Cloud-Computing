## Example: MLP for Non-Linear Classification - XOR Problem

`example_XOR.py` trains a MLP with 2 layers, 2 neurons on each to solve the classic XOR problem, which is a non-linearly separable problem, and thus the reason we need at least 2 layers. Sigmoids are chosen as the hidden activation functions and the output function is a Log-Softmax. The MLP is trained to minimized a Categorical Cross-Entropy performance index.

### New Code

- Using `CrossEntropyLoss` for classification problems, and predicting by taking the label with the highest logit.
- Accessing model's weights and biases to plot simple Decision Boundaries.

## Exercise: 

`exercise_circles.py` asks to apply the same concept on the example to learn a circular decision boundary and visualize it.

### New Code

-  Plotting complex Decision Boundaries using contour plots.