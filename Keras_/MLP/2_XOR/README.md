## Example: MLP for Non-Linear Classification - XOR Problem

`example_XOR.py` trains a MLP with 2 layers, 2 neurons on each to solve the classic XOR problem, which is a non-linearly separable problem, and thus the reason we need at least 2 layers. Sigmoids are chosen as the hidden activation functions and the output function is a softmax. The MLP is trained to minimized a Categorical Cross-Entropy performance index.

### New Code

- One-hot-encoding targets with `keras.utils.to_categorical`.
- Using Categorical Cross-Entropy performance index and accuracy metric on `model.compile`
- Softmax output function for multiclass classification.
- Getting actual predictions using `np.argmax` on the output of `model.predict`.
- Accesing model weights and biases and plotting simple Decision Boundaries.

## Exercise: 

`exercise_circles.py` asks to apply the same concept on the example to classify learn a circular decision boundary and visualize it.

### New Code

-  `model.evaluate` to get the final metrics using the trained model.
-  Plotting complex Decision Boundaries using contour plots.