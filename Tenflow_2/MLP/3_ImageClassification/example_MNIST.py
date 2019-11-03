# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
# TensorFlow 2 runs on GPU by default if you have one, to make sure you can run tf.test.is_gpu_available() and
# doing tf.debugging.set_log_device_placement(True) will tell where each operation is being run
# Sets random seeds for reproducibility. Note that this will not yield fully reproducible results on GPU in most cases
# To know why, watch https://www.youtube.com/watch?v=Ys8ofBeR2kA
tf.random.set_seed(42)
np.random.seed(42)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_NEURONS = (100, 200, 100)
N_EPOCHS = 20
BATCH_SIZE = 128
DROPOUT = 0.2


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
class MLP(tf.keras.Model):
    """ MLP with 3 hidden layers """
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3, dropout=DROPOUT):
        super(MLP, self).__init__()
        self.linear1 = tf.keras.layers.Dense(hidden_dim1, input_shape=(784,))
        # The Batch Normalization normalizes the outputs from the hidden activation functions. This helps with neuron
        self.linear1_bn = tf.keras.layers.BatchNormalization()  # imbalance and can speed training significantly.
        # Note this is an actual layer with some learnable parameters. It's not just min-maxing or standardizing
        self.linear2 = tf.keras.layers.Dense(hidden_dim2)
        self.linear2_bn = tf.keras.layers.BatchNormalization()
        self.linear3 = tf.keras.layers.Dense(hidden_dim3)
        self.linear3_bn = tf.keras.layers.BatchNormalization()
        self.out = tf.keras.layers.Dense(10)
        self.act = tf.nn.relu
        # The dropout is placed right after the outputs of the hidden layers. This sets some of these
        # self.drop = tf.keras.layers.Dropout(dropout)  # outputs to 0, so that a random dropout % of the hidden
        # neurons are not used during each training step, nor are they updated
        # This layer is broken... https://github.com/tensorflow/tensorflow/issues/25175. We will use a function instead
        self.drop = dropout
        self.training = True

    def call(self, x):
        out = tf.nn.dropout(self.linear1_bn(self.act(self.linear1(x)), training=self.training), self.drop)
        out = tf.nn.dropout(self.linear2_bn(self.act(self.linear2(out)), training=self.training), self.drop)
        return self.out(tf.nn.dropout(self.linear3_bn(self.act(self.linear3(out)), training=self.training), self.drop))


# Re-implements the MLP class with an arbitrary number of hidden layers. This way we don't need
# to create another class or modify the existing one if we want to try out more or less layers
class MLPList(tf.keras.Model):
    """ MLP with len(neurons_per_layer) hidden layers """
    def __init__(self, neurons_per_layer, dropout=DROPOUT):
        super(MLPList, self).__init__()
        self.model_layers = [tf.keras.Sequential([tf.keras.layers.Dense(neurons_per_layer[0], input_shape=(784,))])]
        self.model_layers[0].add(tf.keras.layers.ReLU())
        self.model_layers[0].add(tf.keras.layers.BatchNormalization())
        # self.model_layers[0].add(tf.keras.layers.Dropout(dropout))  # This layer is broken...
        # https://github.com/tensorflow/tensorflow/issues/25175  We will use a function instead
        for neurons in neurons_per_layer[1:]:
            self.model_layers.append(
                tf.keras.Sequential([
                    tf.keras.layers.Dense(neurons),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.BatchNormalization()])  # ,
                    # tf.keras.layers.Dropout(dropout)])
            )
        self.model_layers.append(tf.keras.layers.Dense(10))
        self.drop = dropout
        self.training = True

    def call(self, x):
        for layer in self.model_layers[:-1]:
            x = tf.nn.dropout(layer(x, training=self.training), self.drop)
        return self.model_layers[-1](x)


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Reshapes to (n_examples, n_pixels), i.e, each pixel will be an input feature to the model
x_train, x_test = tf.reshape(x_train, (len(x_train), -1)), tf.reshape(x_test, (len(x_test), -1))
x_train, x_test = tf.dtypes.cast(x_train, tf.float32), tf.dtypes.cast(x_test, tf.float32)
y_train, y_test = tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)

# %% ---------------------------------------- Training Prep ------------------------------------------------------------
# Using MLP instead of MLPList will give exactly the same results, as we have the exact same architecture
model = MLP(*N_NEURONS)
# model = MLPList(N_NEURONS)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
@tf.function
def train(x, y):
    model.training = True  # makes BatchNorm use the actual training data to compute the mean and std
    model.drop = DROPOUT  # Activates the dropouts for training
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
    model.training = False  # makes BatchNorm use mean and std estimates computed during training
    model.drop = 0  # Deactivates the dropouts for inference
    # Here we don't use tf.GradientTape()
    logits = model(x)
    loss = criterion(y, logits)
    test_loss(loss)
    test_accuracy(y, logits)


# %% ---------------------------------------- Training Loop ------------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):

    for batch in range(len(x_train)//BATCH_SIZE + 1):  # Loops over the number of batches
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)  # Gets a slice to index the data
        train(x_train[inds], y_train[inds])

    eval(x_test, y_test)

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
    train_loss.reset_states(); train_accuracy.reset_states(); test_loss.reset_states(); test_accuracy.reset_states()

# %% ------------------------------------------- Final Test ------------------------------------------------------------
# Performs the final test on the CPU
with tf.device('/CPU:0'):
    model.training, model.drop = False, 0
    logits = model(x_test)
    print(accuracy_score(y_test.numpy(), tf.argmax(logits, axis=1).numpy())*100)
