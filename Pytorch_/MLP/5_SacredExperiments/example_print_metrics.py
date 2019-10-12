import os
import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_id", default=None, type=int, required=True)
parser.add_argument("--config", default=False, type=bool)
Args = parser.parse_args()


pd.options.display.width = 0
experiments_path = os.getcwd() + "/example_mnist_mlp_runs"
with open(experiments_path + "/{}/metrics.json".format(Args.experiment_id), "r") as s:
    metrics = json.load(s)
print(pd.DataFrame({"training loss": metrics["training loss"]["values"],
                    "training acc": metrics["training acc"]["values"],
                    "testing loss": metrics["testing loss"]["values"],
                    "testing acc": metrics["testing acc"]["values"]}))
if Args.config:
    with open(experiments_path + "/{}/config.json".format(Args.experiment_id), "r") as s:
        config = json.load(s)
    print(config)

# Run python3 example_print_metrics.py --experiment_id 3 on the terminal to print out the training process of run 3
