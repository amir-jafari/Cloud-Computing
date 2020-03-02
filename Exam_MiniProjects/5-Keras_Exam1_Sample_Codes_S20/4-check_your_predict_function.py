# This script shows an example of how to test your predict function before submission
# 0. To use this, replace x_test (line 26) with a list of image paths. And
# 1. Replace predict with your predict function
# or
# 2. Import your predict function from your predict script and remove the predict function define here
# Example: from predict_username (predict_ajafari) import predict
# %% -------------------------------------------------------------------------------------------------------------------
import numpy as np

# This predict is a dummy function
def predict(x):
    # On the exam, x will be a list of all the paths to the images of our held-out set
    images = []
    for img_path in x:
        # Here you would write code to read img_path and preprocess it
        images.append(1)  # I am using 1 as a dummy placeholder instead of the preprocessed image
    x = np.array(images)

    # Here you would load your model and use it on x to get y_pred, and then return y_pred, model
    # (or y_pred, model1, model2, etc.).  # You need to return all the models you used to get y_pred!
    model = lambda p: np.zeros((len(p),))  # Labeling everything as 0 as a dummy placeholder for the model
    y_pred = model(x)
    return y_pred, model

# %% -------------------------------------------------------------------------------------------------------------------
x_test = ["cell_1.png", "cell_2.png", "etc."]  # Dummy image path list placeholder
y_test_pred, *models = predict(x_test)

# %% -------------------------------------------------------------------------------------------------------------------
assert isinstance(y_test_pred, type(np.array([1])))  # Checks if your returned y_test_pred is a NumPy array
assert y_test_pred.shape == (len(x_test),)           # Checks if its shape is this one (one label per image path)
# Checks whether the range of your predicted labels is correct
assert np.unique(y_test_pred).max() <= 3 and np.unique(y_test_pred).min() >= 0
