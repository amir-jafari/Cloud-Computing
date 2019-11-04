# %% -------------------------------------------------------------------------------------------------------------------

# % -----------------------------------------------------------------------------------------------------
# Modify the data prep in order to have a generator instead of loading the whole data into memory at once
# % -----------------------------------------------------------------------------------------------------

# 1. Define a generator function that loads each image and label file and returns the preprocessed tensors.
# Use PIL.Image or other option to read directly from its path and do not depend on TensorFlow path type

# 2. Use tf.data.Dataset.from_generator to get the training and testing datasets from these functions

# % -----------------------------------------------------------------------------------------------------
