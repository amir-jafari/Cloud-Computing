# %% -------------------------------------------------------------------------------------------------------------------

# 1. Imagine you want to experiment with the same model on two datasets, and you want to keep two experiments files.
# Figure out a way to do this by running run_experiment.py only once, i.e, by running this file the model should be
# trained on both datasets, and the results of the experiments for each dataset should be kept track of separately.
# Use mnist and fashionmnist.

# 2. Add an option on main_loop.py so that the best model of each run is saved, instead of saving only the best model
# out of all the runs. The best place to save it is on the folders with the ids for each run.

# 3. Add an option to load a model from a run and continue training after changing some of the hyper-parameters, like
# the learning rate, the optimizer or the number of epochs.

# 4. Create a training file that takes as inputs the dataset and the experiment id to use, loads the corresponding model
# and config, and tests this model on the test set (this makes more sense when we have a held-out set)

# %% -------------------------------------------------------------------------------------------------------------------
