# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.02
N_NEURONS = 2
N_EPOCHS = 5000
PRINT_LOSS_EVERY = 1000


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
class MLP(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.linear1 = tf.keras.layers.Dense(hidden_dim, input_shape=(2,))
        self.act1 = tf.nn.sigmoid
        self.linear2 = tf.keras.layers.Dense(2)

    def call(self, x):
        return self.linear2(self.act1(self.linear1(x)))


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
p = tf.convert_to_tensor(np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]]))
t = tf.convert_to_tensor(np.array([0, 0, 1, 1]))

# %% -------------------------------------- Training Prep --------------------------------------------------------------
model = MLP(N_NEURONS)
optimizer = optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
# A classification problem is usually approached with a Categorical Cross-Entropy performance index
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # This one combines both the loss and the
# Log-SoftMax output function, which computes the probabilities of each example belonging to each class
# This is the preferred way as its computation is more stable. Thus, there is no need to include the
# SoftMax/Log-SoftMax output function on the model itself
# To keep track of the training loss, this will accumulate over each call if not reset
train_loss = tf.keras.metrics.Mean(name='train_loss')
# This decorator converts the eager execution code (Dynamic Graph) into a Static Graph. This means the first time train
# is called on line 60, the graph is built and then when it's called a second time it does not have to be built again
# See https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/ for an in-depth explanation.
# Note that this only allows for debugging on the first forward pass! (The graph is actually being built more than once
# due to TensorFlow internals stuff, but very infrequently). If you put a break point on line 24 and debug, it will
# stop there. If you click Resume Program, it will not stop there on the third forward pass. Now try doing the same
# thing after commenting the decorator (line 48). You will notice the debugger will stop on line 24 every single time.
@tf.function
def train():
    with tf.GradientTape() as tape:
        logits = model(p)
        loss = criterion(t, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


# %% -------------------------------------- Training Loop --------------------------------------------------------------
for epoch in range(N_EPOCHS):
    train()
    if epoch % PRINT_LOSS_EVERY == 0:
        print("Epoch {} | Loss {:.5f}".format(epoch, train_loss.result()))
    train_loss.reset_states()  # Resets the metric for the next epoch

# %% -------------------------------------- Check Approx ---------------------------------------------------------------
print(accuracy_score(t.numpy(), tf.argmax(model(p), axis=1).numpy())*100)

# Gets weights and biases, and from them the DBs, and plots them
w, b = model.linear1.weights[0].numpy(), model.linear1.weights[1].numpy()
a = np.arange(0, 1.1, 0.1)
db1 = -w[0, 0]*a/w[1, 0] - b[0]/w[1, 0]  # Note than on tf.keras the shape of the weight matrix
db2 = -w[0, 1]*a/w[1, 1] - b[1]/w[1, 1]  # is RxS (n_inputs x n_neurons)
p_np = p.numpy()
plt.scatter(p_np[:2, 0], p_np[:2, 1], marker="*", c="b")
plt.scatter(p_np[2:, 0], p_np[2:, 1], marker=".", c="b")
plt.plot(a, db1, c="b")
plt.plot(a, db2, c="b")
plt.show()
