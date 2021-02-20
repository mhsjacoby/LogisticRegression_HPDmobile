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
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, f1_score


def get_counts(df, stage, counts):
    counts[0].append((stage, len(df[df.img == 0])))
    counts[1].append((stage, len(df[df.img == 1])))
    counts[2].append((stage, len(df)))
    return counts


def get_model_metrics(logit_clf, X, y, pred_type):
    """Stand-alone function to get metrics given a classifier.

    Returns: Model probabilities and results mesage for printing
    """

    probs = logit_clf.predict_proba(X)[:,1]

    y_hat = logit_clf.predict(X)
    conf_mat = pd.DataFrame(confusion_matrix(y_hat, y), 
                            columns = ['Vacant', 'Occupied'],
                            index = ['Vacant', 'Occupied']
                            )

    conf_mat = pd.concat([conf_mat], keys=['Actual'], axis=0)
    conf_mat = pd.concat([conf_mat], keys=['Predicted'], axis=1)
    print(pred_type, 'Confusion Matrix')
    print(conf_mat)

    logging.info(f'\n{conf_mat}')

    score = logit_clf.score(X, y)
    RMSE = np.sqrt(mean_squared_error(y, y_hat))
    f1 = f1_score(y, y_hat)

    results_msg = f'{pred_type} results on {len(X)} predictions\n'\
                    f'\tScore: {score:.4}\n' \
                    f'\tRMSE: {RMSE:.4}\n' \
                    f'\tF1-score: {f1:.4}'

    logging.info(results_msg)
    return (probs, results_msg)


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
        

        # print(f'Fill type {self.fill_type}')
        # print(f'images 0s {len(df[df.img == 0])}')
        # print(f'images 1s {len(df[df.img == 1])}')
        # print(f'total length {len(df)}')


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

        
        log_config = logging.basicConfig(
            filename=os.path.join(self.log_save_dir, f'{log_type}_{home}.log'),
            level=logging.INFO,
            format='%(message)s',
            datefmt='%Y-%m-%d',
            )
        log_obj = logging.getLogger(log_config)
        log_obj.info(f'\n\t\t##### NEW RUN {log_type} #####\n{date.today()} -- {datetime.now().strftime("%H:%M")}')
        return log_obj


def create_lags(df, lag_hours=8, min_inc=5):
    """Creates lagged occupancy variable

    Takes in a df and makes lags up to (and including) lag_hours.
    The df is in 5 minute increments (by default), so lag is 12*hour.
    
    Returns: lagged df
    """
    ts = int(60/min_inc)
    # occ_series = df['occupied']
    # logging.info(f'Creating data with a lag of {lag_hours} hours.')
    # self.etl_log.info(f'Creating data with a lag of {lag_hours} hours.')


    for i in range(1, lag_hours+1):
        lag_name = f'lag{i}_occupied'
        df[lag_name] = occ_series.shift(periods=ts*i)
    return df


# now = datetime.now()

# current_time = now.strftime("%H:%M:%S")
# print("Current Time =", current_time)
