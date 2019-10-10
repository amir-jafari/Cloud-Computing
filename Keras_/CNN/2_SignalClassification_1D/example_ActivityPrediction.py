# Download WISDM_ar_latest.tar.gz clicking on "Download Latest Version" at http://www.cis.fordham.edu/wisdm/dataset.php
# Unzip it in the current working directory

# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-4
N_EPOCHS = 15
BATCH_SIZE = 128
DROPOUT = 0.5
DATA_PATH = os.getcwd() + "/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
data_raw = pd.read_csv(DATA_PATH, error_bad_lines=False, header=None)
data_raw.dropna(inplace=True)  # There is only one NaN
data_raw[5] = data_raw[5].apply(lambda f: float(f[:-1]))  # Fixes last column (z-axis)
groups = data_raw.groupby(1)  # Groups by type of activity (classes)
# Loops over the data to get signals per class
x, y, min_signal_length = [], [], 10000000
for label, group in groups:
    for user in np.unique(group[0]):
        x.append(group[group[0] == user][[3, 4, 5]].values)
        y.append(label)
        if min_signal_length > len(x[-1]):
            min_signal_length = len(x[-1])
# Uses the min_signal_length to get fixed length signals, but this means losing some data
x_prep, y_prep = [], []
for signal, label in zip(x, y):
    for i in range(len(signal)//min_signal_length):  # Losing some data here
        x_prep.append(signal[min_signal_length*i:min_signal_length*(i+1)])
        y_prep.append(label)
x, y = np.array(x_prep), np.array(y_prep)
del x_prep, y_prep
print(np.unique(y, return_counts=True))
# Final usual pre-processing
le = LabelEncoder()
y = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2, stratify=y)
sd = StandardScaler()
y_train, y_test = to_categorical(y_train, num_classes=6), to_categorical(y_test, num_classes=6)
# x_train and x_test are already on the suitable shape (batch, steps, channels)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Sequential([  # min_signal_length turns out to be 544, so we pick 4 1D-convs of 544/4 kernel size each
    Conv1D(16, 136, activation="relu"),  # output (batch, 409, 16)
    BatchNormalization(),
    Conv1D(32, 136, activation="relu"),  # output (batch, 274, 32)
    BatchNormalization(),
    Conv1D(64, 136, activation="relu"),  # output (batch, 139, 64)
    BatchNormalization(),
    Conv1D(128, 136, activation="relu"),  # output (batch, 4, 128)
    BatchNormalization(),
    Flatten(),  # output (batch, 512)
    Dense(136, activation="relu"),  # Intermediate linear layer that comes back to the kernel size
    Dropout(DROPOUT),
    BatchNormalization(),
    Dense(6, activation="softmax")
])
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
y_test_pred = np.argmax(model.predict(x_test), axis=1)
print(le.inverse_transform(np.array([0, 1, 2, 3, 4, 5])))
print(confusion_matrix(np.argmax(y_test, axis=1), y_test_pred))
