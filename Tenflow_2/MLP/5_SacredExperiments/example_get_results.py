import os
import json
import pandas as pd


def get_results():
    experiments_path = os.getcwd() + "/example_mnist_mlp_runs"
    # Loads the first experiment and makes the dictionary keys as lists
    with open(experiments_path + "/1/info.json", "r") as s:
        results = json.load(s)
    with open(experiments_path + "/1/config.json", "r") as s:
        config = json.load(s)
    for key in results.keys():
        results[key] = [results[key]]
    for key in config.keys():
        config[key] = [config[key]]
    experiment_ids = [1]
    # Appends the results and configs from the rest of the experiments to these lists
    count = 1
    for experiment in sorted([int(i) for i in os.listdir(experiments_path) if i not in ["_sources", ".DS_Store", "1"]]):
        try:
            with open(experiments_path + "/" + str(experiment) + "/info.json", "r") as s:
                rresults = json.load(s)
            with open(experiments_path + "/" + str(experiment) + "/config.json", "r") as s:
                cconfig = json.load(s)
            for key in rresults.keys():
                results[key].append(rresults[key])
            # Accounts for new hyper-parameters added after having run some experiments already
            for key in cconfig.keys():
                if key in config:
                    config[key].append(cconfig[key])
                else:
                    config[key] = ["-"]*count + [cconfig[key]]
        except:
            pass
        experiment_ids.append(experiment)
        count += 1
    # Saves to spreadsheet after sorting by smallest loss
    a = pd.DataFrame({"experiment_id": experiment_ids}).join(pd.DataFrame(results)).join(pd.DataFrame(config))
    a.sort_values(by=["test loss"], inplace=True, ascending=True)
    a.drop(["n_epochs", "random_seed", "seed"], axis=1, inplace=True)
    a.to_excel(os.getcwd() + "/example_experiments.xlsx")
