import numpy as np
from keras.datasets import mnist

# This script generates the training set (the one you will be provided with),
# and the held out set (the one we will use to test your model towards the leaderboard).

(x_train, y_train), (x_test, y_test) = mnist.load_data()
np.save("x_train.npy", x_train); np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test); np.save("y_test.npy", y_test)
