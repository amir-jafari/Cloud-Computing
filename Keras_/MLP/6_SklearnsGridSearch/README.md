## Example: Energy Efficiency

This example illustrates how to use `sklearn.model_selection.GridSearchCV` with Keras' Sequential models, which also works for other classes like `RandomizedSearchCV` and `StratifiedKFold`. It fits a MLP to estimate two energy-efficiency related target variables of buildings based on achirecture features.

### New Code

- Using `KerasRegressor` for compatibility with sklearn.
- Using `GridSearchCV` for model selection and saving all the results to a spreadsheet, including the refitted best final model.
- Saving the best model.


## Exercise: Bank Marketing

Extends the example to use the best set of hyper-parameters to train a regular Keras model via Early Stopping. It also allows for early stopping during the grid search.

### New Code

- Using `KerasClassifier`.
- Using different metrics on the GridSearch to get a variaty of models that best fit the business purpose. 
- Refitting model with the best hyper-parameters via early-stopping using a dev set, and getting the final metrics on the held-out set (test set).
- Performing early-stopping during the grid search and setting a patience value.