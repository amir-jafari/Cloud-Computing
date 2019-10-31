# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
# It is always good practice to set the hyper-parameters at the beginning of the script
# And even better to define a params class if the script is long and complex
LR = 2.5e-1
N_NEURONS = 10
N_EPOCHS = 50000
PRINT_LOSS_EVERY = 1000


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
# Defines model class. Inherits from tf.keras.Model to get all useful methods and make the class compatible with tf
class MLP(tf.keras.Model):
    """ MLP with 2 layers, sigmoid transfer function and hidden_dim neurons on the hidden layer"""
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()  # Initializes tf.keras.Model
        # Linear Layer that maps input (n_examples, 1) to hidden_dim (number of neurons)
        self.linear1 = tf.keras.layers.Dense(hidden_dim, input_shape=(1,))
        self.act1 = tf.nn.sigmoid  # Hidden transfer function
        # Linear Layer that maps hidden_dim to output (n_examples, 1), i.e, 1 neuron
        self.linear2 = tf.keras.layers.Dense(1)

    # Calling this method "call" will make it possible to go forward by just calling model(x) (see below)
    def call(self, x):
        # Just a sequential pass to go through the 2 layers
        return self.linear2(self.act1(self.linear1(x)))


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Gets the input interval and the targets
p = np.linspace(-2, 2, 100)  # (100,)
t = np.exp(p) - np.sin(2*np.pi*p)  # (100,)
# Converts to Tensors and reshapes to suitable shape (n_examples, 1)
p = tf.reshape(tf.convert_to_tensor(p), (-1, 1))
t = tf.reshape(tf.convert_to_tensor(t), (-1, 1))

# %% -------------------------------------- Training Prep --------------------------------------------------------------
# Initializes model
model = MLP(N_NEURONS)
# Initializes a Gradient Descent Optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=LR)
# Initializes a Mean Square Error performance index
criterion = tf.keras.losses.MeanSquaredError()

# %% -------------------------------------- Training Loop --------------------------------------------------------------
# Starts the training loop
for epoch in range(N_EPOCHS):
    # All operations using tf.Variables (the parameters of the model) are recorded in the Computational Graph
    with tf.GradientTape() as tape:
        t_pred = model(p)
        loss = criterion(t, t_pred)
    # Computes the gradients using the recorded operations (i.e, goes backwards through the Computational Graph)
    gradients = tape.gradient(loss, model.trainable_variables)
    # Updates the parameters
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Checks the training process
    if epoch % PRINT_LOSS_EVERY == 0:
        print("Epoch {} | Loss {:.5f}".format(epoch, tf.reduce_mean(loss).numpy()))

# %% -------------------------------------- Check Approx ---------------------------------------------------------------
# Visually shows the approximation obtained by the MLP
plt.title("MLP fit to $y = e^x - sin(2 \\pi x)$ | MSE: {:.5f}".format(tf.reduce_mean(loss).numpy()))
plt.xlabel("x")
plt.ylabel("y")
plt.plot(p.numpy(), t.numpy(), label="Real Function")
plt.plot(p.numpy(), t_pred.numpy(), linestyle="dashed", label="MLP Approximation")
plt.legend()
plt.show()
