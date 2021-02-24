# Logistic Regression Data Pipeline
Author: Maggie Jacoby
Last update: 2021-02-24

This codebase consists of 4 complete scripts right now. 

### data_basics.py

This script contains helper functions:
- `get_predictions_wGT `
- `get_model_metrics`

and the class `ModelBasics`, which is the parent class for the rest of the classes in the codebase.

`ModelBasics` has 3 functions:
- `get_directories` returns the names of all storage directoires used in the code.
- `read_config` reads a passed configuration file, or finds the appropriate one for the home.
It then returns the configurations, which can be accessed for the class through `self.configs`.
- `format_logs` starts a new logging session (if it's being called form the top-level program), 
or amends the current session.