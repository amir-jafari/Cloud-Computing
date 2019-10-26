from sacred.observers import FileStorageObserver
from example_get_results import get_results
from example_main_loop import ex

# Creates a simple observer or loads an existing one. Will generate example_mnist_mlp_runs if the folder does not exist
ex.observers.append(FileStorageObserver.create('example_mnist_mlp_runs'))
# Runs one experiment. Will generate a folder inside example_mnist_mlp_runs with the name equal to the experiment id
ex.run(config_updates={"lr": 1e-4,
                       }
       )
get_results()  # Gets the results after the experiment has finished
# Runs another two experiments
ex.run(config_updates={"neurons_per_layer": (300, 100),
                       "batch_size": 256
                       }
       )
get_results()
ex.run(config_updates={"dropout": 0.4,
                       }
       )
get_results()
