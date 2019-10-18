# %% -------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# Listen to the outputs of CNN layers that learn local features about songs regarding their genre
# ------------------------------------------------------------------------------------------------

# 1. Re-implement the network from the example, but this time using Model (see MLP/4_FunctionalAPI), and give each
# layer a distinct name. You can use a for loop to write all the CNN architecture in a few of lines

# 2. Train the model and save it

# 3. Define a function that takes as input an example, the model, the layer name, the path to save the file,
# the channel id and an option to average all the channels of the intermediate output, and that saves the output of
# such layer to a .wav file. You can use scipy.io.wavfile.write

# 4. Add an option to only load a saved model, do so and play around with the function defined in 3.

# %% -------------------------------------------------------------------------------------------------------------------
