# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
from time import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, KFold

# Download ENB2012_data.xlsx from https://archive.ics.uci.edu/ml/datasets/energy+efficiency to the current directory

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
# The GridSearchCV with 5 folds and this hyper-parameters takes ...
LR = [5e-4]
N_NEURONS = [(100, 200, 100), (300, 200, 100)]
N_EPOCHS = [2000]
BATCH_SIZE = [512]
DROPOUT = [0.2, 0.3]
ACTIVATION = ["tanh", "relu"]

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
data = pd.read_excel("ENB2012_data.xlsx")  # Reads the dataset into a pandas DataFrame
data.replace("?", np.NaN, inplace=True)  # UCI's nans sometimes come like this
x, y = data.drop(["Y1", "Y2"], axis=1), data[["Y1", "Y2"]]  # Features and target
assert np.all(x.isna().sum() == 0), "There are missing feature values"  # Checks if there are
assert np.all(y.isna().sum() == 0), "There are missing target values"  # any nans
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.3, random_state=SEED)


# %% -------------------------------------- Training Prep --------------------------------------------------------------
# Defines a function to return the compiled model for each set of hyper-parameters on the grid search
def construct_model(dropout=0.3,
                    activation='tanh',
                    n_neurons=(100, 200, 100),
                    lr=1e-3
                    ):

    mlp = Sequential([
        Dense(n_neurons[0], input_dim=8, activation=activation),
        Dropout(dropout),
        BatchNormalization()
    ])
    for neurons in n_neurons[1:]:
        mlp.add(Dense(neurons, activation=activation))
        mlp.add(Dropout(dropout, seed=SEED))
        mlp.add(BatchNormalization())
    mlp.add(Dense(2))  # We have two continious targets, so 2 neurons on output layer
    mlp.compile(optimizer=Adam(lr=lr), loss="mean_squared_error")

    return mlp


model = GridSearchCV(
    estimator=KerasRegressor(
        build_fn=construct_model
    ),
    scoring="r2",
    param_grid={
        'epochs': N_EPOCHS,  # The param grid must contain arguments of the usual Keras model.fit
        "batch_size": BATCH_SIZE,  # and/or the arguments of the construct_model function
        'dropout': DROPOUT,
        "activation": ACTIVATION,
        'n_neurons': N_NEURONS,
        'lr': LR,
    },
    n_jobs=1,
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
    verbose=10
)

# %% -------------------------------------- Training Loop ----------------------------------------------------------
start = time()
model.fit(x_train, y_train)
print(time() - start)

print("The best parameters are:", model.best_params_)
# Gets the results into a DataFrame, drops some "irrelevant" columns, sorts by best score and saves to spreadsheet
results = pd.DataFrame(model.cv_results_).drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
                                                'params', 'std_test_score', 'rank_test_score'], axis=1)
results.drop(list(results.columns[results.columns.str.contains("split")]), axis=1, inplace=True)
results.sort_values(by="mean_test_score", ascending=False).to_excel("example_experiments.xlsx")
# Saves the best model
print("Saving refitted best model on the whole training set...")
model.best_estimator_.model.save("mlp_energy.hdf5")

# %% ------------------------------------------ Final test -------------------------------------------------------------
# Gets final score on the test set using the best model refitted (done by default in model.fit) on the whole train set
print("Final test r-squared:", model.score(x_test, y_test))
# from sklearn.metrics import r2_score
# print(r2_score(y_test, model.best_estimator_.predict(x_test)))
