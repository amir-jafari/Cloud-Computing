# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import tensorflow as tf


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 5e-2
N_EPOCHS = 30
BATCH_SIZE = 512
DROPOUT = 0.5


# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, 3)  # output (n_examples, 26, 26, 16)
        self.convnorm1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(2)  # output (n_examples, 13, 13, 16)
        self.conv2 = tf.keras.layers.Conv2D(32, 3)  # output (n_examples, 11, 11, 32)
        self.convnorm2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.AveragePooling2D(2)  # output (n_examples, 5, 5, 32)
        self.flatten = tf.keras.layers.Flatten()  # input will be flattened to (n_examples, 32 * 5 * 5)
        self.linear1 = tf.keras.layers.Dense(400)
        self.linear1_bn = tf.keras.layers.BatchNormalization()
        self.linear2 = tf.keras.layers.Dense(10)
        self.act = tf.nn.relu
        self.drop = DROPOUT
        self.training = True

    def call(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x)), training=self.training))
        x = self.flatten(self.pool2(self.convnorm2(self.act(self.conv2(x)), training=self.training)))
        x = tf.nn.dropout(self.linear1_bn(self.act(self.linear1(x)), training=self.training), self.drop)
        return self.linear2(x)


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Reshapes to (n_examples, height, width, n_channels)
x_train, x_test = tf.reshape(x_train, (len(x_train), 28, 28, 1), ), tf.reshape(x_test, (len(x_test), 28, 28, 1))
x_train, x_test = tf.dtypes.cast(x_train, tf.float32), tf.dtypes.cast(x_test, tf.float32)
y_train, y_test = tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN()
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


# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):

    for batch in range(len(x_train)//BATCH_SIZE + 1):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        train(x_train[inds], y_train[inds])

    eval(x_test, y_test)

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
    train_loss.reset_states(); train_accuracy.reset_states(); test_loss.reset_states(); test_accuracy.reset_states()
