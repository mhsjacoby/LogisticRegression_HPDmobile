# Logistic Regression Data Pipeline
Author: Maggie Jacoby  
Last update: 2021-02-24

This codebase consists of five complete scripts right now. 

## data_basics.py
This is imported in all subsequent scripts. It contains helper functions and the parent class
(`ModelBasics`) for the rest of the classes in the code base.  

### Functions
- `get_predictions_wGT ` takes a logistic classifier and X data and makes predictions using the 
*ground truth* for the lags (not previous predictions). This is used to report results in training,
and for comparing with the *results using past predictions* in testing.
- `get_model_metrics` takes in numpy lists of ground truth (`y_true`) and predictions (`y_hat`)
and generates a variety of 

`ModelBasics` **functions**:
- `get_directories` returns the names of all storage directoires used in the code.
- `read_config` reads a passed configuration file, or finds the appropriate one for the home.
It then returns the configurations, which can be accessed for the class through `self.configs`.
- `format_logs` starts a new logging session (if it's being called form the top-level program), 
or amends the current session.
___

## etl.py
This class ("extract, transform, load") processes all data for training and testing.  
Initialize it with:
- The fill type: `zeros` (default), `ones`, or `ffill`
- The home (e.g. `H1`)

#### When initialized:
This program loads the data type, which can be `train`, `test`, or `train and test` (default).
- If the requested type exists, it reads the csv file.
- If the requested type DOESN'T exist, it reads in the inferences for that type, fills nans,
creates lags, and writes both train and test csvs (using the dates provided in the config file).
- In both cases, it stores the requested type in `self.train` and `self.test`
- This class also contains the function `split_xy`, which is used when the data is called in the
`train.py` and `test.py` programs.

TODO: Create function to read inferences from different hubs and combine for house level train/test data
___

## train.py
This class trains and returns a logistic regression model.

- Initialize with the same fill type and home as etl.py

- The configuration file specifies model parameters (see `H1_train_config.yaml`).

- You can initialize it with training data (`X_train`, `y_train`), or it uses `ETL` class to load 
and split the requested fill type. 

- The LR classifier is saved under `self.model`.

- Optionally, you can give a new model save name, otherwise it increments a new model based on the 
number of previous models for the home. Models are written to .pickle files.
___

## test.py
This class tests the trained model. 

- Initialize with the same fill type and home as etl.py

- You can initialize it with a model object, or it will read one in from a pickle file.
You can specify the model file to read in, or it will choose one. 

- The function `test_model` runs tests using the ground truth lags AND using the predictions.

- Results from both are saved with the object, but only predictions are stored in the logfiles. 
Results are calculated with the `get_model_metrics` function from `data_basics.py` and `additional_metrics`
function (native to this class). Results are formatted with `print_results`.

- a CSV file is saved that contains the predictions (0/1), the probability (within 0 to 1),
and the ground truth.
___

## run_train_test.py
This file combines the above three class to create train/test datafile, train a model, and test it.
This is the easiest way to run the above programs. 

Run the program from the terminal like this:
`$ python3 run_train_test.py`
Optional arguments: 
`-train_home`  (default is H1)
`-test_home` (if not specified, uses train home)  
`-fill_type` (`zeros` (default), `ones`, or `ffill`)

#TODO: Write loop that trains/tests multiple homes/sensors
___

## TODO
Other scripts to write:
- `plot_predictions.py`: Read in csv results and create beautiful graphs to show performance.

- `explore.py`: Perform model selection and corss validation to find the BEST model for each home.  
