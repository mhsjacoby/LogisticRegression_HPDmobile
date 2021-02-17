"""
data_basics.py
Author: Maggie Jacoby
Last update: 2021-02-17
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
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix


def get_model_metrics(logit_clf, X, y, pred_type='Train'):
    """Stand-alone function to get metrics given a classifier.

    Prints and logs metrics.
    Returns: Model predictions
    """

    probs = logit_clf.predict(X)
    y_hat = pd.Series(probs).apply(lambda x: 1 if (x > 0.5) else 0)

    conf_mat = pd.DataFrame(confusion_matrix(y_hat, y), 
                            columns = ['Vacant', 'Occupied'],
                            index = ['Vacant', 'Occupied']
                            )

    conf_mat = pd.concat([conf_mat], keys=['Actual'], axis=0)
    conf_mat = pd.concat([conf_mat], keys=['Predicted'], axis=1)

    logging.info(f'\n{conf_mat}')

    score = logit_clf.score(X, y)
    RMSE = np.sqrt(mean_squared_error(y, y_hat))

    results_msg = f'\n{pred_type} Results on {len(X)} predictions\n'\
                    f'\tScore: {score:.4}\n' \
                    f'\tRMSE: {RMSE:.4}'

    logging.info(results_msg)
    print(results_msg)
    return probs


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

    def read_config(self, config_files, config_type='ETL'):
        """Reads in the configuration file (*.yaml).
        
        Returns: configuration parameters
        """
        if len(config_files) == 0:
            print(f'No {config_type} configuration file for {self.home}. Exiting program.')
            logging.info('No configuration file.')
            sys.exit()

        config_file_path = config_files[0]
        logging.info(f'{len(config_files)} {config_type} configuration file(s).\
                    \nUsing: {os.path.basename(config_file_path)}')

        with open(config_file_path) as f:
            config = yaml.safe_load(f)
        return config

    def format_logs(self, log_type, home):
        """Creates log object.

        Returns: nothing
        """
        os.makedirs(self.log_save_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(self.log_save_dir, f'{log_type}_{home}.log'),
            level=logging.INFO,
            format='%(message)s',
            datefmt='%Y-%m-%d',
            )
        logging.info(f'\n\t\t##### NEW RUN #####\n{date.today()}')



