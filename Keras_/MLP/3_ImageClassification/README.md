## Example: MLP for Image Classification - MNIST Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is the Iris dataset of Neural Networks when it comes to images. It contains 60,000 training and 10,000 testing examples of labaled hand-written grayscale digits from 0 to 9. Each image is 28x28 pixels, but these will be flattened into 768 dimensions, which will serve as features for the MLP.

`example_MNIST.py` trains a MLP with 3 layers, dropout, batch normalization and hidden relu activations. $R = 768$, $S_M = 10$. The output function is a Log-Softmax, and the MLP is trained to minimized a Categorical Cross-Entropy performance index.

### New Code

- "Reproducibility".
- Loading data from `keras.datasets`.
- Dropouts and Batch Normalization.
- Using `model.add` on a sequential model to add an arbitraty number of hidden layers in a for loop.
- Mini-batching and evaluating on validation set during training on `model.fit`.


## Exercise: MLP for Image Classification - FashionMNIST Dataset

`exercise_FashionMNIST.py` asks to train a MLP on the [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, which is exactly the same as the MNIST dataset but replacing the digits with images of different types of clothing. It was made to have a harder benchmark for image classification models. The adaptation from `example_MNIST.py` is thus trivial, although the MLP accuracy is 10% lower than on the MNIST dataset.

### New Code

- Early Stopping
- Loading and Saving Models
- Plotting images 
