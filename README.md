# Logistic Regression Data Pipeline
Author: Maggie Jacoby  
Last update: 2021-02-24

This codebase consists of 5 complete scripts right now. 

## data_basics.py
This is imported in all subsequent scripts. It contains helper functions and the parent class
(`ModelBasics`) for the rest of the classes in the code base.  

### Functions
- `get_predictions_wGT ` takes a logistic classifier and X data and makes predictions using the 
*ground truth* for the lags (not previous predictions). This is used to report results in training,
and for comparing with the *results using past predictions* in testing.
- `get_model_metrics` takes in numpy lists of ground truth (`y_true`) and predictions (`y_hat`)
and generates a variety of 

`ModelBasics` **has 3 functions**:
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