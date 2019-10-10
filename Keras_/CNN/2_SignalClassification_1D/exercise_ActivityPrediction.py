# %% -------------------------------------------------------------------------------------------------------------------

# % ----------------------------------------------------------------------------------
# Improve the pre-processing and the model to recognize human activity in a better way
# % ----------------------------------------------------------------------------------

# 1. The previous pre-processing was very naive because the model was trained on very long sequences, even when it was
# the shortest sequence length in our dataset. A good model should be able to recognize human activity right after the
# user starts doing such activity. WISDM_ar_v1.1_raw_about.txt states that the time step is 50 milliseconds.
# If we want the model to tell us what the user has started to do after say 3 seconds, it should work for input signals
# of 3/0.05 = 60 time steps.
# Define a hyper-parameter called MIN_SIGNAL_DUR that is equal to this number of seconds and pre-process accordingly.

# 2. The previous model will not work with this new input sequence length. It could also be argued that the convolutions
# had kernel sizes that were too big, compared to the usual kernel size in the literature. Modify the CNN & Dense layers
# to work with the new input shape and train the model. Follow the same pattern we used on the example and compare.

# 3. Even the kernel size from 2. could be considered too big. Modify the model from 2. to use kernels size of 3. Due to
# this, we would need some pooling in-between CNN layers in order not to have a very huge model. Also, replace the
# penultimate dense layer and the flatten operation with a Global Average Pooling layer.
# Refer to 1_ImageClassification/exercise_solution_FashionMNIST for an example. Train the model and compare with 2.

# %% -------------------------------------------------------------------------------------------------------------------
