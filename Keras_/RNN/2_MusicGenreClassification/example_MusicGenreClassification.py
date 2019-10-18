# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, LSTM, Dense, MaxPooling1D
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix

if "genres" not in os.listdir(os.getcwd()):
    try:
        os.system("wget http://opihi.cs.uvic.ca/sound/genres.tar.gz")
        os.system("tar -xvzf genres.tar.gz")
    except:
        print("There was an error trying to download the data!")
        raise
        # Go to http://marsyas.info/downloads/datasets.html and Download the GTZAN genre collection (Approximately 1.2GB)
    if "genres" not in os.listdir(os.getcwd()):
        print("There was an error trying to download the data!")
        import sys
        sys.exit()

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
DATA_PATH = "/genres"

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 50
BATCH_SIZE = 16
DROPOUT = 0.5
SEQ_LEN = 10  # seconds
HIDDEN_SIZES = [256, 128]

# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def get_max_length():
    max_length = 0
    for subdir in [f for f in os.listdir(os.getcwd() + DATA_PATH) if os.path.isdir(os.getcwd() + DATA_PATH + "/" + f)]:
        for file in os.listdir(os.getcwd() + DATA_PATH + "/" + subdir):
            _, example = wavfile.read(os.getcwd() + DATA_PATH + "/" + subdir + "/" + file)
            if len(example) > max_length:
                max_length = len(example)
    return max_length

def load_data():
    x, y, label, label_dict = [], [], 0, {}
    for subdir in [f for f in os.listdir(os.getcwd() + DATA_PATH) if os.path.isdir(os.getcwd() + DATA_PATH + "/" + f)]:
        label_dict[subdir] = label
        for file in os.listdir(os.getcwd() + DATA_PATH + "/" + subdir):
            _, example = wavfile.read(os.getcwd() + DATA_PATH + "/" + subdir + "/" + file)
            if len(example) > 22050*SEQ_LEN:  # Trims from the beginning to max_seq_length
                example = example[:22050*SEQ_LEN]  # 22050 is the sampling frequency (22050 samples per second)
            else:  # Pads up to the max_seq_length
                example = np.hstack((example, np.zeros((22050*SEQ_LEN - len(example)))))
            x.append(example)
            y.append(label)
        label += 1
    return np.array(x), np.array(y), label_dict

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
if SEQ_LEN == "get_max_from_data":
    SEQ_LEN = get_max_length()//22050  # In seconds
x, y, labels = load_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=SEED, stratify=y)
# Reshapes to (batch_size, timesteps, input_dim). We have only one channel (input_dim) due to mono audio files
x_train, x_test = x_train.reshape(len(x_train), -1, 1), x_test.reshape(len(x_test), -1, 1)
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model1 = Sequential([
    Conv1D(8, 3, activation="relu"),
    MaxPooling1D(4),
    BatchNormalization(),
    Conv1D(16, 3, activation="relu"),
    MaxPooling1D(4),
    BatchNormalization(),
    Conv1D(32, 3, activation="relu"),
    MaxPooling1D(4),
    BatchNormalization(),
    Conv1D(64, 3, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(4),
    Conv1D(128, 3, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(4),
    Conv1D(256, 3, activation="relu"),
    MaxPooling1D(4),
    BatchNormalization(),
    # MaxPooling1D(4),
])
HIDDEN_SIZES[0] = 256
for hidden_size in HIDDEN_SIZES[:-1]:  # If we want more than one LSTM layer, we need to return the sequences on all the layers
    model1.add(LSTM(units=hidden_size, dropout=DROPOUT, recurrent_dropout=DROPOUT, return_sequences=True))  # except for the last one
model1.add(LSTM(units=HIDDEN_SIZES[-1], dropout=DROPOUT, recurrent_dropout=DROPOUT))
model1.add(Dense(10, activation="softmax"))
model1.compile(optimizer=RMSprop(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
model1.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
           callbacks=[ModelCheckpoint("example_cnn_lstm_music_genre_classifier.hdf5", monitor="val_accuracy", save_best_only=True)])

# %% ------------------------------------------ Final Test -------------------------------------------------------------
model = load_model('example_cnn_lstm_music_genre_classifier.hdf5')
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print(labels)
print("The confusion matrix is:")
print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1)))
