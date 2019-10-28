# %% -------------------------------------------------------------------------------------------------------------------

# % ----------------------------------------------------------
# Train a 1D-CNN to tell whether two sentences are paraphrases
# % ----------------------------------------------------------

# 1. Download the Microsoft Research Paraphrase Corpus from https://gluebenchmark.com/tasks. For mac and linux users,
# go to https://github.com/wasiahmad/paraphrase_identification/tree/master/dataset/msr-paraphrase-corpus for the .txt.

# 2. Replace the acc function with an equivalent that returns the f1 score and use this metric for early stopping.

# 3. Write a function to load the data from the .txt files. The usual way of pre-processing is stacking both sentences
# together and separating them by a "SEP" special token. The two sentences constitute one input.

# 4. Modify the CNN class so that it works with sentences of 80 words (this should be the maximum sequence length
# extracted from the corpus).

# %% -------------------------------------------------------------------------------------------------------------------
