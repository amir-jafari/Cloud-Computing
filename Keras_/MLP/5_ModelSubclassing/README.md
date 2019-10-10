## Example: Keras Model Subclassing

This example illustrates how to use Model subclassing instead of the Sequential class or the functional API. This allows more flexibility on the design than the functional API and is very similar to PyTorch. However, some of the functionaility is lost.

### New Code

- Defining own model class by inheriting from `Model`.


## Exercise: Keras Model Subclassing loss of functionality

Asks to mimic `4_FunctionalAPI\exercise_FashionMNIST.py`. It will be noticed that the way of accesing intermidiate outputs needs to change.

### New Code

- Getting intermediate outputs from a subclassed model by using tensorflow backend explicitly.
- NOTE: This works using TF1 as backend, but I haven't been able to make it work with TF2.
