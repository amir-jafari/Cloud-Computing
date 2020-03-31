# This script shows an example of how to test your predict function before submission
# 0. To use this, replace x_test (line 26) with a list of absolute image paths. And
# 1. Replace predict with your predict function
# or
# 2. Import your predict function from your predict script and remove the predict function define here
# Example: from predict_username (predict_ajafari) import predict
# %% -------------------------------------------------------------------------------------------------------------------
import numpy as np
import torch

# This predict is a dummy function, yours can be however you like as long as it returns the predictions in the right format
def predict(x):
    # On the exam, x will be a list of all the paths to the images of our held-out set
    images = []
    for img_path in x:
        # Here you would write code to read img_path and preprocess it
        images.append(1)  # I am using 1 as a dummy placeholder instead of the preprocessed image
    x = torch.FloatTensor(np.array(images))

    # Here you would load your model (.pt) and use it on x to get y_pred, and then return y_pred
    model = lambda p: torch.ones((len(p), 7))  # Labeling everything with all the labels as a dummy placeholder for the model
    y_pred = model(x)
    return y_pred

# %% -------------------------------------------------------------------------------------------------------------------
x_test = ["/home/ubuntu/data/test/cells_1.png", "/home/ubuntu/data/test/cells_2.png", "etc."]  # Dummy image path list placeholder
y_test_pred = predict(x_test)

# %% -------------------------------------------------------------------------------------------------------------------
assert isinstance(y_test_pred, type(torch.Tensor([1])))  # Checks if your returned y_test_pred is a Torch Tensor
assert y_test_pred.dtype == torch.float  # Checks if your tensor is of type float
assert y_test_pred.device.type == "cpu"  # Checks if your tensor is on CPU
assert y_test_pred.requires_grad is False  # Checks if your tensor is detached from the graph
assert y_test_pred.shape == (len(x_test), 7)  # Checks if its shape is the right one
# Checks whether the your predicted labels are one-hot-encoded
assert set(list(np.unique(y_test_pred))) in [{0}, {1}, {0, 1}]
print("All tests passed!")
