## Example: Keras functional API

This example illustrates how to use the functional API instead of the Sequential class. This allows to control the inputs at each layer of the model, reuse the same layer for different inputs and some other convinient functionality (see https://keras.io/getting-started/functional-api-guide/)

### New Code

- `Input` and `Model` classes in the context of the functional API.
- Controlling the inputs to each layer and merging inputs from different layers.


## Exercise: Keras functional API functionality

`exercise_FashionMNIST.py` asks to exploit the functional API to improve accuracy on this dataset, compared to the Sequential model. It also asks to retrieve intermidiate outputs of the model and vizualize them.

### New Code

- Getting intermediate outputs of a model.
