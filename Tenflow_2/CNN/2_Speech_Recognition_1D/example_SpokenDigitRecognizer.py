# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import numpy as np
import tensorflow as tf
from shutil import copyfile
from scipy.io import wavfile
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

if "free-spoken-digit-dataset-master" not in os.listdir(os.getcwd()):
    try:
        os.system("wget https://github.com/Jakobovski/free-spoken-digit-dataset/archive/master.zip")
        os.system("unzip master.zip")
    except:
        print("There was a problem downloading the data!")
        raise
    if "free-spoken-digit-dataset-master" not in os.listdir(os.getcwd()):
        print("Please download the data")
        import sys
        sys.exit()
    # The recordings folder can be found at https://github.com/Jakobovski/free-spoken-digit-dataset
    # You can clone the repo into the current working directory

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

DATA_PATH = os.getcwd() + "/free-spoken-digit-dataset-master/recordings/"
DATA_PATH_TRAIN = os.getcwd() + "/train_data/"
DATA_PATH_TEST = os.getcwd() + "/test_data/"
TRAIN = True
SAVE_MODEL = False
SAVE_MODEL_PATH = "example_saved_model/"

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 50
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 1  # (In seconds)
# MAX_SEQ_LENGTH = "get_from_data"

# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def make_train_test_dirs(data_path, dir_train, dir_test, overwrite=False):
    if (not os.path.isdir(dir_train) and not os.path.isdir(dir_test)) or overwrite:
        os.mkdir(dir_train); os.mkdir(dir_test)
        for name in ["jackson", "nicolas", "theo", "yweweler"]:
            for digit in range(10):
                for example_id in range(50):
                    file_name = "{}_{}_{}.wav".format(digit, name, example_id)
                    if example_id >= 5:
                        copyfile(data_path + file_name, dir_train + "/" + file_name)
                    else:
                        copyfile(data_path + file_name, dir_test + "/" + file_name)

def get_max_length():
    """ Gets the length of the longest audio file """
    max_length, all_files = 0, filter(lambda f: ".wav" in f, os.listdir(DATA_PATH))
    all_files = [DATA_PATH + f for f in all_files]
    for file in all_files:
        _, example = wavfile.read(file)
        if len(example) > max_length:
            max_length = len(example)
    return max_length

def load_data(data_path, max_seq_length=1):
    """ Loads all the training or testing data into two big tensors """
    x, y, files = [], [], filter(lambda f: ".wav" in f, os.listdir(data_path))
    for file in [data_path + f for f in files]:
        _, example = wavfile.read(file)
        if len(example) > 8000*max_seq_length:  # Trims from the beginning to max_seq_length
            example = example[:8000*max_seq_length]  # 8000 is the sampling frequency (8000 samples per second)
        else:  # Pads up to the max_seq_length
            example = np.hstack((example, np.zeros((8000*max_seq_length - len(example)))))
        x.append(example.reshape(-1, 1))  # Reshapes to (width, n_channels)
        y.append(int(file[file.find("data/") + 5]))
    return tf.dtypes.cast(tf.convert_to_tensor(x), tf.float32), tf.convert_to_tensor(np.array(y))

# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(16, 30)  # Slides 16 kernels of size 30 across the temporal dimension
        self.convnorm1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv1D(32, 30)
        self.convnorm2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool1D(2)  # The max pooling is the main source of dim reduction in this case

        self.conv3 = tf.keras.layers.Conv1D(64, 30)
        self.convnorm3 = tf.keras.layers.BatchNormalization()

        self.conv4 = tf.keras.layers.Conv1D(128, 30)
        self.convnorm4 = tf.keras.layers.BatchNormalization()
        self.pool4 = tf.keras.layers.MaxPool1D(2)

        self.conv5 = tf.keras.layers.Conv1D(256, 30)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.linear = tf.keras.layers.Dense(10)

        self.act = tf.nn.relu
        self.training = True

    def call(self, x):
        x = self.convnorm1(self.act(self.conv1(x)), training=self.training)
        x = self.pool2(self.convnorm2(self.act(self.conv2(x)), training=self.training))
        x = self.convnorm3(self.act(self.conv3(x)), training=self.training)
        x = self.pool4(self.convnorm4(self.act(self.conv4(x)), training=self.training))
        return self.linear(self.global_avg_pool(self.act(self.conv5(x))))

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
make_train_test_dirs(DATA_PATH, DATA_PATH_TRAIN, DATA_PATH_TEST)
if MAX_SEQ_LENGTH == "get_from_data":
    MAX_SEQ_LENGTH = get_max_length()
if TRAIN:
    x_train, y_train = load_data(DATA_PATH_TRAIN, max_seq_length=MAX_SEQ_LENGTH)
x_test, y_test = load_data(DATA_PATH_TEST, max_seq_length=MAX_SEQ_LENGTH)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN()
if TRAIN:
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    # @tf.function
    def train(x, y):
        model.training = True
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = criterion(y, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(y, logits)

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    # @tf.function
    def eval(x, y):
        model.training = False
        logits = model(tf.dtypes.cast(tf.convert_to_tensor(x), tf.float32))
        loss = criterion(y, logits)
        test_loss(loss)
        test_accuracy(y, logits)

# %% ---------------------------------------- Training Loop ------------------------------------------------------------
if TRAIN:
    loss_test_best = 1000
    print("Starting training loop...")
    for epoch in range(N_EPOCHS):
        # Initiates a progress bar that will be updated for each batch. # "Epoch" will be updated for each epoch
        with tqdm(total=len(x_train)//BATCH_SIZE + 1, desc="Epoch {}".format(epoch)) as pbar:
            for batch in range(len(x_train)//BATCH_SIZE + 1):
                inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
                train(x_train[inds], y_train[inds])
                pbar.update(1)  # Updates the progress and the training loss
                pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss.result()))

        eval(x_test, y_test)

        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))

        if test_loss.result().numpy() < loss_test_best and SAVE_MODEL:
            model_path = SAVE_MODEL_PATH + "cnn_spoken_digit_recognizer"
            model.save_weights(model_path, save_format='tf')
            print("The model has been saved!")
            loss_test_best = test_loss.result().numpy()

        train_loss.reset_states(); train_accuracy.reset_states(); test_loss.reset_states(); test_accuracy.reset_states()

# %% ----------------------------------------- Final Test --------------------------------------------------------------
model_path = SAVE_MODEL_PATH + "cnn_spoken_digit_recognizer"
model.load_weights(model_path)
model.training = False
y_test_pred = tf.argmax(model(x_test), axis=1).numpy()
print("The accuracy on the test set is", 100*accuracy_score(y_test.numpy(), y_test_pred), "%")
print("The confusion matrix is")
print(confusion_matrix(y_test.numpy(), y_test_pred))
