from example_get_results import get_results
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sacred import Experiment


# Creates a experiment, or loads an existing one
ex = Experiment('example_mnist_mlp')


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
# Creates a hyper-parameter config that will be passed on the main function and that can be changed when running the
# experiment (see example_run_experiment.py)
@ex.config
def my_config():
    random_seed = 42
    lr = 1e-3
    neurons_per_layer = (100, 200, 100)
    n_epochs = 2
    batch_size = 512
    dropout = 0.2


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
# You can define functions and classes outside of the main function
class MLP(tf.keras.Model):
    """ MLP with len(neurons_per_layer) hidden layers """
    def __init__(self, neurons_per_layer, dropout):
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
        self.model_layers.append(tf.keras.layers.Dense(10))
        self.drop = dropout
        self.training = True

    def call(self, x):
        for layer in self.model_layers[:-1]:
            x = tf.nn.dropout(layer(x, training=self.training), self.drop)
        return self.model_layers[-1](x)



@ex.automain
def my_main(random_seed, lr, neurons_per_layer, n_epochs, batch_size, dropout):

    # %% --------------------------------------- Set-Up ----------------------------------------------------------------
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    # %% -------------------------------------- Data Prep --------------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = tf.reshape(x_train, (len(x_train), -1)), tf.reshape(x_test, (len(x_test), -1))
    x_train, x_test = tf.dtypes.cast(x_train, tf.float32), tf.dtypes.cast(x_test, tf.float32)
    y_train, y_test = tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)

    # %% -------------------------------------- Training Prep ----------------------------------------------------------
    model = MLP(neurons_per_layer, dropout)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    @tf.function
    def train(x, y):
        model.training = True
        model.drop = dropout
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
    print("\n ------------ Doing run number {} with configuration ---------------".format(ex.current_run._id))
    print(ex.current_run.config)
    try:  # Gets the best result so far, so that we only save the model if the result is better (test loss in this case)
        get_results()
        results_so_far = pd.read_excel(os.getcwd() + "/example_experiments.xlsx")
        loss_test_best = min(results_so_far["test loss"].values)
    except Exception as e:
        print(e)
        loss_test_best = 1000
        print("No results so far, will save the best model out of this run")
    best_epoch, loss_best, acc_best = 0, 1000, 0

    print("Starting training loop...")
    for epoch in range(n_epochs):

        for batch in range(len(x_train) // batch_size + 1):
            inds = slice(batch * batch_size, (batch + 1) * batch_size)
            train(x_train[inds], y_train[inds])

        eval(x_test, y_test)
        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, train_loss.result(), train_accuracy.result() * 100, test_loss.result(),
                                        test_accuracy.result() * 100))

        # Only saves the model if it's better than the models from all of the other experiments
        if test_loss.result().numpy() < loss_test_best:
            tf.saved_model.save(model, os.getcwd() + '/example_mlp_mnist/')
            print("A new model has been saved!")
            loss_test_best = test_loss.result().numpy()
        if test_loss.result().numpy() < loss_best:
            best_epoch, loss_best, acc_best = epoch, test_loss.result().numpy(), test_accuracy.result().numpy()

        # To keep track of the metrics during the training process on metrics.json
        ex.log_scalar("training loss", train_loss.result().numpy(), epoch)
        ex.log_scalar("training acc", train_accuracy.result().numpy(), epoch)
        ex.log_scalar("testing loss", test_loss.result().numpy(), epoch)
        ex.log_scalar("testing acc", test_accuracy.result().numpy(), epoch)
        # To save the best results of this run to info.json. This is used by get_results() to generate the spreadsheet
        ex.info["epoch"], ex.info["test loss"], ex.info["test acc"] = best_epoch, loss_best, acc_best

        train_loss.reset_states(); train_accuracy.reset_states(); test_loss.reset_states(); test_accuracy.reset_states()
