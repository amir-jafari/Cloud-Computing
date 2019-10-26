from example_get_results import get_results
import os
import torch
import torch.nn as nn
from torchvision import datasets
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
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


# %% ----------------------------------- Helper Functions --------------------------------------------------------------
# You can define functions and classes outside of the main function
def acc(model, x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, neurons_per_layer, dropout):
        super(MLP, self).__init__()
        dims = (784, *neurons_per_layer)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(dims[i+1]),
                nn.Dropout(dropout)
            ) for i in range(len(dims)-1)
        ])
        self.layers.extend(nn.ModuleList([nn.Linear(neurons_per_layer[-1], 10)]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@ex.automain
def my_main(random_seed, lr, neurons_per_layer, n_epochs, batch_size, dropout):

    # %% --------------------------------------- Set-Up ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # %% -------------------------------------- Data Prep --------------------------------------------------------------
    data_train = datasets.MNIST(root='.', train=True, download=True)
    x_train, y_train = data_train.data.view(len(data_train), -1).float().to(device), data_train.targets.to(device)
    x_train.requires_grad = True
    data_test = datasets.MNIST(root='.', train=False, download=True)
    x_test, y_test = data_test.data.view(len(data_test), -1).float().to(device), data_test.targets.to(device)

    # %% -------------------------------------- Training Prep ----------------------------------------------------------
    model = MLP(neurons_per_layer, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # %% -------------------------------------- Training Loop ----------------------------------------------------------
    print("\n ------------ Doing run number {} with configuration ---------------".format(ex.current_run._id))
    print(ex.current_run.config)
    try:  # Gets the best result so far, so that we only save the model if the result is better (test loss in this case)
        get_results()
        results_so_far = pd.read_excel(os.getcwd() + "/example_experiments.xlsx")
        loss_test_best = min(results_so_far["test loss"].values)
    except:
        loss_test_best = 1000
        print("No results so far, will save the best model out of this run")
    best_epoch, loss_best, acc_best = 0, 1000, 0

    print("Starting training loop...")
    for epoch in range(n_epochs):

        loss_train = 0
        model.train()
        for batch in range(len(x_train) // batch_size + 1):
            inds = slice(batch * batch_size, (batch + 1) * batch_size)
            optimizer.zero_grad()
            logits = model(x_train[inds])
            loss = criterion(logits, y_train[inds])
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        model.eval()
        with torch.no_grad():
            y_test_pred = model(x_test)
            loss = criterion(y_test_pred, y_test)
            loss_test = loss.item()

        acc_train, acc_test = acc(model, x_train, y_train), acc(model, x_test, y_test)
        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, loss_train/batch_size, acc_train, loss_test, acc_test))

        # Only saves the model if it's better than the models from all of the other experiments
        if loss_test < loss_test_best:
            torch.save(model.state_dict(), "mlp_mnist.pt")
            print("A new model has been saved!")
            loss_test_best = loss_test
        if loss_test < loss_best:
            best_epoch, loss_best, acc_best = epoch, loss_test, acc_test

        # To keep track of the metrics during the training process on metrics.json
        ex.log_scalar("training loss", loss_train/batch_size, epoch)
        ex.log_scalar("training acc", acc_train, epoch)
        ex.log_scalar("testing loss", loss_test, epoch)
        ex.log_scalar("testing acc", acc_test, epoch)
        # To save the best results of this run to info.json. This is used by get_results() to generate the spreadsheet
        ex.info["epoch"], ex.info["test loss"], ex.info["test acc"] = best_epoch, loss_best, acc_best

    # sleep(5)  # In case the above line takes a bit...
    #     try:
    #         get_results()
    #         print("Results so far have been saved!")
    #     except Exception as e:
    #         print(e)
    #         print("There was an error saving the results for this run...")
