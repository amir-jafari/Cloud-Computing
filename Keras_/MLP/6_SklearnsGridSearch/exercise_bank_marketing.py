# %% -------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# Train a MLP to predict whether a customer will subscribe to a bank after a pre-defined phone call.
# --------------------------------------------------------------------------------------------------

# 1. Download bank-additional.zip from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing, unzip it and move
# bank-additional-full.csv to the current directory.

# 2. Choose the hyper-parameters values you want to test on the GridSearchCV. You should manually try different learning
# rates and number of epochs at least, to narrow down the search before doing the grid search.

# 3. Read the data and handle missing values and do whatever other pre-processing you want after some optional EDA.

# 4. Define the function that will return the MLP and also the GridSearchCV instance that uses KerasClassifier, and fit.
# Think about which metric you should use for GridSearchCV, although without more info we can't really make a good
# decision in this case. We would need data on the money the bank makes out of each customer they sign up and the amount
# of money it spends on people calling customers.

# 5. After you are content with a cross-val performance on the train set, get the final accuracy on the test set.

# 6. Use the best model's hyperparameters to get the model again, but this time as a regular Sequential model
# (you can use the construct_model function). Split the train set into train and dev, use the train set to train this
# model and perform early stopping on the dev set. Then get the final performance on the test set and compare with 5.

# 7. Repeat 4. and 5. but this time performing EarlyStopping during the grid search. Use some patience value so that the
# training process is stopped if the val loss does not decrease for some epochs. After that, do 6. with the new best set
# of hyper-parameteres. Compare the two new final test scores with each other and with 5. and 6. Hint: Follow
# https://stackoverflow.com/questions/48127550/early-stopping-with-keras-and-sklearn-gridsearchcv-cross-validation

# %% -------------------------------------------------------------------------------------------------------------------
