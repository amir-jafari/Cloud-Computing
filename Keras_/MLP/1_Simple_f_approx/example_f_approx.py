# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
# It is always good practice to set the hyper-parameters at the beginning of the script
# And even better to define a params class if the script is long and complex
LR = 0.1
N_NEURONS = 10
N_EPOCHS = 50000

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Gets the input interval and the targets
p = np.linspace(-2, 2, 100)  # (100,)
t = np.exp(p) - np.sin(2*np.pi*p)  # (100,)

# %% -------------------------------------- Training Prep --------------------------------------------------------------
# Defines the model
model = Sequential([
    Dense(N_NEURONS, input_dim=1),  # Linear layer that maps 1 input dim to N_NEURONS hidden dim
    Activation("sigmoid"),  # Sigmoid hidden transfer function
    Dense(1)  # Maps N_NEURONS hidden dim to 1 output dim
])
# Prepares a Stochastic Gradient Descent optimizer and a Mean Squared Error performance index
model.compile(optimizer=SGD(lr=LR), loss="mean_squared_error")

# %% -------------------------------------- Training Loop --------------------------------------------------------------
# Trains the model. We use full Batch GD.
train_hist = model.fit(p, t, epochs=N_EPOCHS, batch_size=len(p))

# %% -------------------------------------- Check Approx ---------------------------------------------------------------
# Visually shows the approximation obtained by the MLP
plt.title("MLP fit to $y = e^x - sin(2 \\pi x)$ | MSE: {:.5f}".format(train_hist.history["loss"][-1]))
plt.xlabel("x")
plt.ylabel("y")
plt.plot(p, t, label="Real Function")
plt.plot(p, model.predict(p), linestyle="dashed", label="MLP Approximation")
plt.legend()
plt.show()
