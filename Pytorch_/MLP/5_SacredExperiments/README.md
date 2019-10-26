## Example: Keeping track of your experiments using Sacred

While PyTorch is a very flexible, easy-to-use and powerful framework for ANNs, usually a lot of experiments are needed in order to find a good network for your problem. It can be really hard to keep track of all the things you have tried, and what lead to improved results and what not. Here comes [sacred](https://sacred.readthedocs.io/en/latest/index.html). This examples shows how to keep records of all your experiments in a simple spreadsheet by using this package.

There is a main file with the training code on `example_main_loop.py`. The hyper-parameters to this file can be controled with the file `example_run_experiment.py`, which is the file you want to run. `example_get_results.py` defines a function to save the best result of each experiment into a spreadsheet, and `example_print_metrics.py` provides an command line interface to print out the training process of an experiment.

### New Code

- Working with more than one script and making imports between them.
- Using sacred's `Experiment`, `@ex.config` and `@ex.automain` decorators, `ex.log_scalar` to keep track of training metrics and `ex.info` as an intermidiate step to save results into a spreadsheet.
- Using `argparse` to control arguments to a script from the command line.

## Exercise: Adding more advanced functionalities

This exercise extends the example to allow for some usuful additional features, which require more advanced Python knowledge.

### New Code

- Deleting imported scripts from `sys.modules`.
- Importing on script B from the script A that runs script B (`__main__`)
- Using `locals().update(my_dict)` to change global variables using a dictionary.
