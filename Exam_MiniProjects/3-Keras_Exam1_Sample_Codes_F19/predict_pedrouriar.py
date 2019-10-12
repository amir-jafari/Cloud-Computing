# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
from keras.models import load_model


def predict(x):
    # Here x is a NumPy array. On the actual exam it will be a list of paths.
    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    x = x.reshape(len(x), -1)
    x = x / 255
    # Write any data prep you used during training
    # %% --------------------------------------------- Predict ---------------------------------------------------------
    model = load_model('mlp_pedrouriar.hdf5')
    y_pred = np.argmax(model.predict(x), axis=1)
    return y_pred
