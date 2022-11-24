### Code structure
The file part2_house_value_regression.py contains all the necessary functions to train the model and perform hyperparameter tuning. The added functions are:
1. `example_main` trains a regressor model and prints the training error. To change the regressor parameters, you can change lines 634-637.
2. `RegressorHyperParameterSearch` splits the dataset into train/validation and performs hyperparameter tuning over the specified search space.
3. The functions `depth_tuning`,`neurons_tuning`,`lr_tuning`, and `batch_size_tuning` were used to find the optimal hyperparameters, implementing the the top-down approach detailed in our report. We ran these functions sequentially to find the optimal depth, number of neurons, learning rate and batch size. The final model is obtained by running the `batch_size_tuning` function.

### Training a Regressor model
To train a Regressor model, simply run `python part2_house_value_regression.py`. This will run the function `example_main`, which will train a Regressor model and print the training error. To train a Regressor model with different parameters, change lines 675-677:
```
    regressor = Regressor(
        x_train, neurons = [32, 32, 32, 1], activations = ['relu', 'relu', 'relu', 'relu'], 
        learning_rate = 0.001, nb_epoch = 2000, batch_size=32)
```

### Performing hyperparameter tuning
To test any of the tuning functions, uncomment the corresponding line from lines 692-698. Each function performs a search over a specific hyperparamater, and saves a dictionary with the evaluation results used in the report as `results_HYPERPARAMETER.pkl`. It also saves the best model as `part2_model_tuned_HYPERPARAMETER.pickle`. 

The best model is saved as `part2_model.pickle`. It was obtained using the `batch_size_tuning` function, because batch_size is the last hyperparameter we tune, so the function already uses the optimal depth, number of neurons and learning rate found in previous steps.
