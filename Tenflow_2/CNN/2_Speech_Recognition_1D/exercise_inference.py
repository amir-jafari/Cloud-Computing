# %% -------------------------------------------------------------------------------------------------------------------

# % -------------------------------------------------------------------------------------
# Use the model from the example to make inference on self-recorded spoken digits
# % -------------------------------------------------------------------------------------

# Note: If you want to record yourself, download Audacity (https://www.audacityteam.org/download/), and select
# "Project Rate (Hz)" = 8000, and "1 (Mono) Recording Channel" before making the recordings

# 1. Define a function to load the "zero.wav", "one.wav", etc. files and preprocess it to input into the model

# 2. Define a function to take as input the pre-processed tensor and return the predicted label and probabilities

# 3. Re-define the model class from the example

# 4. Load the model and set it up for inference

# 5. Print out the real labels, and the predicted labels and predicted probabilities of each label by the model

# %% -------------------------------------------------------------------------------------------------------------------
