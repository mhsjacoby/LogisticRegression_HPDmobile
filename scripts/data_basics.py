"""
data_basics.py
Author: Maggie Jacoby
Last update: 2021-02-17

TODO:
- Make logging function more robust (add multiple file handlers) to log to multiple locations per session

"""

import os
import sys
import csv
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, date
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, f1_score, accuracy_score


def get_predictions_wGT(logit_clf, X, y):
    """Run data through classifier to get predictions given X and y using ground truth for lags

    Returns: probabilities (between 0,1) and predictions (0/1)
    """
    probs = logit_clf.predict_proba(X)[:,1]
    preds = logit_clf.predict(X)

    # conf_mat, results = get_model_metrics(y_true=y, y_hat=preds)
    # logging.info(f'\n=== TRAINING RESULTS === \n\n{conf_mat}')
    return probs, preds


def get_model_metrics(y_true, y_hat):
    """Stand-alone function to get metrics given a classifier.

    Returns: Confusion matrix and list of results as tuples
    """
    conf_mat = pd.DataFrame(confusion_matrix(y_hat, y_true), 
                            columns = ['Vacant', 'Occupied'],
                            index = ['Vacant', 'Occupied']
                            )
    conf_mat = pd.concat([conf_mat], keys=['Actual'], axis=0)
    conf_mat = pd.concat([conf_mat], keys=['Predicted'], axis=1)

    score = accuracy_score(y_true, y_hat)
    RMSE = np.sqrt(mean_squared_error(y_true, y_hat))
    f1 = f1_score(y_true, y_hat)

    results_metrics = [
                        ('length', len(y_true)),
                        ('Accuracy', f'{score:.4}'),
                        ('RMSE', f'{RMSE:.4}'),
                        ('F1', f'{f1:.4}')
                        ]

    return conf_mat, results_metrics


class ModelBasics():
    """Parent class for ETL, TrainModel, and TestModel.
    
    Contains basic functions for logging, getting configuration files, 
    getting storage directory locations, and getting list of days.
    """ 

    def get_directories(self):
        """Gets names of all directories, relative to the current script.

        Returns: nothing
        """
        parent_dir = os.path.dirname(os.getcwd())

        self.config_dir = os.path.join(parent_dir, 'configuration_files')
        self.log_save_dir = os.path.join(parent_dir, 'logs')
        self.data_dir = os.path.join(parent_dir, 'data')
        self.models_dir = os.path.join(parent_dir, 'models')
        self.raw_data = os.path.join(parent_dir, 'raw_data_files')
        self.result_csvs = os.path.join(parent_dir, 'result_csvs')


    def read_config(self, config_files, config_type='ETL'):
        """Reads in the configuration file (*.yaml).
        
        Returns: configuration parameters
        """
        if len(config_files) == 0:
            print(f'No {config_type} configuration file for {self.home}. Exiting program.')
            sys.exit()

        config_file_path = config_files[0]
        logging.info(f'{len(config_files)} {config_type} configuration file(s).\
                    \nUsing: {os.path.basename(config_file_path)}')

        with open(config_file_path) as f:
            config = yaml.safe_load(f)
        return config


    def format_logs(self, log_type, home):
        """Creates log object and set logging parameters and format

        Returns: log object
        """
        os.makedirs(self.log_save_dir, exist_ok=True)

        log_config = logging.basicConfig(
            filename=os.path.join(self.log_save_dir, f'{home}.log'),
            level=logging.INFO,
            format='%(message)s',
            datefmt='%Y-%m-%d',
            )
        log_obj = logging.getLogger(log_config)
        log_obj.info(f'\n\t\t##### {log_type} #####\n{date.today()}: {datetime.now().strftime("%H:%M")}')
        return log_obj




