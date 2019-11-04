# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
from shutil import copyfile
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

if "jpg" and "imagelabels.mat" in os.listdir(os.getcwd()):
    pass
else:
    try:
        os.system("wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz")
        os.system("wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat")
        os.system("tar -xvzf 102flowers.tgz")
    except:
        print("There was an error downloading the data!")
        raise
        # Go to http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html and click "Dataset images" and "The image labels"
        # Unzip 102flowers.tgz on the current working directory and move imagelabels.mat also in there
    if "jpg" and "imagelabels.mat" in os.listdir(os.getcwd()):
        pass
    else:
        print("There was an error downloading the data!")
        import sys
        sys.exit()

# This example uses tf.Dataset.from_tensor_slices, which means we need to use TF methods to load and manipulate the data
# This is not ideal (which will be solved in the exercise by using a generator), but still very useful.

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
DATA_DIR = "/jpg/"
TRAIN_TEST_SPLIT = True

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-2
N_NEURONS = (100, 200, 100)
N_EPOCHS = 100
BATCH_SIZE = 512
DROPOUT = 0.5


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
class MLP(tf.keras.Model):
    """ MLP with len(neurons_per_layer) hidden layers """
    def __init__(self, neurons_per_layer, dropout=DROPOUT):
        super(MLP, self).__init__()
        self.model_layers = [tf.keras.Sequential([tf.keras.layers.Dense(neurons_per_layer[0], input_shape=(784,))])]
        self.model_layers[0].add(tf.keras.layers.ReLU())
        self.model_layers[0].add(tf.keras.layers.BatchNormalization())
        for neurons in neurons_per_layer[1:]:
            self.model_layers.append(
                tf.keras.Sequential([
                    tf.keras.layers.Dense(neurons),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.BatchNormalization()])
            )
        self.model_layers.append(tf.keras.layers.Dense(102))
        self.drop = dropout
        self.training = True

    def call(self, x):
        for layer in self.model_layers[:-1]:
            x = tf.nn.dropout(layer(x, training=self.training), self.drop)
        return self.model_layers[-1](x)


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
if TRAIN_TEST_SPLIT:
    if not os.path.isdir("data"):
        os.mkdir("data")
        os.mkdir("data/train"); os.mkdir("data/test")
        x = np.array([i for i in range(1, 8190)])  # Creates example vector with ids corresponding to the file names
        y = loadmat("imagelabels.mat")["labels"].reshape(-1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.3, stratify=y)
        for example, label in zip(x_train, y_train):
            id_path = "0"*(5-len(str(example))) + str(example)  # For each example, gets the actual id of the file
            copyfile(os.getcwd() + DATA_DIR + "image_{}.jpg".format(id_path),
                     os.getcwd() + "/data/train/" + "image_{}.jpg".format(id_path))
            with open(os.getcwd() + "/data/train/" + "image_{}.txt".format(id_path), "w") as s:
                s.write(str(label-1))  # labels start at 1 --> 1 to 0, 2 to 1, etc.
        for example, label in zip(x_test, y_test):
            id_path = "0"*(5-len(str(example))) + str(example)
            copyfile(os.getcwd() + DATA_DIR + "image_{}.jpg".format(id_path),
                     os.getcwd() + "/data/test/" + "image_{}.jpg".format(id_path))
            with open(os.getcwd() + "/data/test/" + "image_{}.txt".format(id_path), "w") as s:
                s.write(str(label-1))
    else:
        print("The data split dir is not empty! Do you want to overwrite it?")


# This function will be used to map the raw files dataset to an actual dataset of tensors
def load_prep_image(image_path):  # We need to use the TensorFlow way of reading images because we are using
    img = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=1)  # td.data.Dataset.from_tensor_slices
    return tf.reshape(tf.image.resize(img, [28, 28]), (784,))


# Gets all the paths for the training images and their labels
train_paths_x = [os.getcwd() + "/data/train/" + path for path in os.listdir("data/train") if path[-4:] == ".jpg"]
train_y = [os.getcwd() + "/data/train/" + path for path in os.listdir("data/train") if path[-4:] == ".txt"]
for idx in range(len(train_y)):  # Gets the actual labels as integers
    with open(train_y[idx], "r") as s:
        train_y[idx] = int(s.read())
# Creates a Dataset made up of the training paths
path_ds = tf.data.Dataset.from_tensor_slices(train_paths_x)
# New dataset made up of the TensorFlow images. num_parallel_calls to process elements in parallel
image_ds = path_ds.map(load_prep_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(train_y)  # For the labels we already have integers
ds_train = tf.data.Dataset.zip((image_ds, label_ds))  # This is like: for image, label in zip(x_train, y_train)
# Randomly shuffles the dataset. buffer_size must be equal or greater than the number of training examples
ds_train = ds_train.shuffle(buffer_size=len(train_paths_x))
ds_train = ds_train.batch(BATCH_SIZE)  # batches the dataset
# `prefetch` lets the dataset fetch batches in the background while the model is training.
ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Repeats for testing but without shuffling and getting only one batch (the whole thing)
test_paths_x = [os.getcwd() + "/data/test/" + path for path in os.listdir("data/test") if path[-4:] == ".jpg"]
test_paths_y = [os.getcwd() + "/data/test/" + path for path in os.listdir("data/test") if path[-4:] == ".txt"]
for idx in range(len(test_paths_y)):
    with open(test_paths_y[idx], "r") as s:
        test_paths_y[idx] = int(s.read())
path_ds = tf.data.Dataset.from_tensor_slices(test_paths_x)
image_ds = path_ds.map(load_prep_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(test_paths_y)
ds_test = tf.data.Dataset.zip((image_ds, label_ds))
ds_test = ds_test.batch(len(test_paths_x))

# %% ---------------------------------------- Training Prep ------------------------------------------------------------
model = MLP(N_NEURONS)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
@tf.function
def train(x, y):
    model.training = True
    model.drop = DROPOUT
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = criterion(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(y, logits)

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
@tf.function
def eval(x, y):
    model.training = False
    model.drop = 0
    logits = model(x)
    loss = criterion(y, logits)
    test_loss(loss)
    test_accuracy(y, logits)


# %% ---------------------------------------- Training Loop ------------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):

    for image_batch, label_batch in ds_train:
        train(image_batch, label_batch)
    # ds_train = ds_train.unbatch()  # Uncomment this if you want to shuffle the whole thing again
    ds_train = ds_train.shuffle(buffer_size=len(train_paths_x))  # We shuffle again after each epoch
    # ds_train = ds_train.batch(BATCH_SIZE)  # Uncomment this to work with ds_train.unbatch()
    # ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Idem, although not really needed
    # Uncommenting the commented lines above will shuffle all the elements of the dataset on each epoch
    # Right now, what will be shuffled on each are the batches, but the elements on each batch will remain the same

    for image_batch, label_batch in ds_test:  # Although this is a for loop, it will only be one iteration
        eval(image_batch, label_batch)

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
    train_loss.reset_states(); train_accuracy.reset_states(); test_loss.reset_states(); test_accuracy.reset_states()
