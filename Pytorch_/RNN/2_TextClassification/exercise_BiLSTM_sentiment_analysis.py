# %% -------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------
# Use Bidirectional LSTMs for Sentiment Analysis
# ----------------------------------------------

# 1. Modify the code from the example to use a bidirectional lstm. You will need to play around with the output of this.
# Below are some of the things you can try:

# 2. Add the outputs of each LSTM element-wise on the hidden dimension before passing them to the mean layer from the example.

# 3. Concat the outputs of each LSTM on the hidden dimension and then pass them to the mean layer (which needs more neurons).

# 4. Use a mean layer like on the example for each of the LSTM outputs and then concat.

# 5. Concat the outputs of each LSTM on the hidden dimension but now don't use a mean layer, just use the last output
# of the LSTMs and input it to a final classification layer.

# 6. Repeat 5 but before passing this last output to the classif layer, use a pooling layer to aggregate all the time steps.
# This can be done using a linear layer with input_dim = hidden_dim


# %% -------------------------------------------------------------------------------------------------------------------
