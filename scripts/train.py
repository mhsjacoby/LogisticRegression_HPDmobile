"""
train.py
Authors: Maggie Jacoby and Jasmine Garland
Last update: 2021-02-17
"""

import os
import sys
import csv
import yaml
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, date

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix

from data_basics import ModelBasics, get_model_metrics, get_counts
from etl import ETL 


class TrainModel(ModelBasics):
    """Trains a logistic regression model.

    Uses ETL to load the training data.
    Writes a pickle file with the trained LR model at the end.
    """

    def __init__(self, home, fill_type, save_model=False, 
                save_fname=None, config_file=None, X_train=None, y_train=None):

        self.fill_type = fill_type
        self.counts = counts
        self.home = home
        self.get_directories()
        self.coeff_msg = None
        self.predicted_probabilities = None
        self.results_msg = None

        self.train_log = self.format_logs(log_type='Train', home=self.home)
        config_file_list = self.pick_config_file(config_file)
        self.configs = self.read_config(config_files=config_file_list, config_type='Train')

        if not X_train:
            self.X, self.y = self.get_train_data()
        self.model = self.train_model()
        
        if save_model:
            self.save_model(model=self.model, model_name=save_fname)#, overwrite=overwrite)


    def pick_config_file(self, config_file=None):
        """
        """
        if not config_file:
            config_file_list = glob(os.path.join(self.config_dir, f'{self.home}_train_*.yaml'))
        else:
            # print(f'>>> Configuration file specified: {config_file}.')
            config_file_list = glob(os.path.join(self.config_dir, config_file))
        
        return config_file_list


    def get_train_data(self):
        """Imports training data from ETL class

        Returns: training dataset
        """

        self.train_log.info('Parameters used:')
        for param in self.configs:
            self.train_log.info(f'\t{param}: {self.configs[param]}')        

        Data = ETL(self.home, data_type='train', fill_type=self.fill_type, counts=self.counts)
        X, y = Data.split_xy(Data.train)
        return X, y


    def set_LR_parameters(self):
        """Sets the model parameters as specified in the configuration file.

        Only takes in one set of parameters.
        Returns: sklearn logistic regression model object (not fitted to data)
        """
        clf = LogisticRegression().set_params(**self.configs)
        return clf

############
    def train_model(self):
        """Trains a logistic regression model.

        Uses default parameters, or gets params from ParameterGrid (if specified).
        Returns: sklearn logistic regression model object
        """
        logit_clf = self.set_LR_parameters()
        X = self.X.to_numpy()
        y = self.y.to_numpy()
        logit_clf.fit(X, y)

        self.train_probs = logit_clf.predict_proba(X)[:,1]
        self.train_preds = logit_clf.predict(X)

        conf_mat, results = get_model_metrics(y_true=y, y_hat=self.train_preds)
        # self.predicted_probabilities, self.results_msg = results

        self.coeff_msg = self.print_coeffs(logit_clf)
        logging.info(f'{self.coeff_msg}')
        # self.train_probs, self.train_preds = probs

        return logit_clf


    def print_coeffs(self, model):

        coeff_msg = f'\nCoefficients:\n{pd.Series(model.coef_[0], index = self.X.columns).to_string()}\n' \
                        f'intercept\t{model.intercept_[0]}'
        return coeff_msg


    def get_filename(self, model_save_dir):
        """Gets model number if filename not specified.

        Increments name based on total number of models.
        Params:

        Returns: filename to save model as 
        """
        models = glob(os.path.join(model_save_dir, '*.pickle'))
        if not models:
            fname = f'{self.home}_model.pickle'
        else:
            model_num = len(models) + 1
            fname = f'{self.home}_model_{model_num}.pickle'
        return fname


    def save_model(self, model, model_name=None, overwrite=False):
        """Saves model as a pickle object. 

        Names model with filename (if specified) or numerically. 
        Returns: nothing
        """
        model_save_dir = os.path.join(self.models_dir, self.home)
        os.makedirs(model_save_dir, exist_ok=True)

        if not model_name:
            fname = self.get_filename(model_save_dir=model_save_dir)
        else:
            fname = f'{model_name}.pickle'
        save_name = os.path.join(model_save_dir, fname)

        if not os.path.isfile(save_name):
            pickle.dump(model, open(save_name, 'wb'))
            print(f'>>> Saving model to {fname}')
            logging.info(f'Writing model to {fname}.')
        else:
            pickle.dump(model, open(save_name, 'wb'))
            print(f'>>> Model {fname} exists. Overwriting previous.')
            logging.info(f'Overwriting previous {fname}.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and save models')
    parser.add_argument('-home', '--home', default='H1', type=str, help='Home to get data for, eg H1')
    parser.add_argument('-save_fname', '--save_fname', default=None, help='Filename to save model to')
    parser.add_argument('-config_file', '--config_file', default=None, help='Configuration file to use')
    parser.add_argument('-fill_type', '--fill_type')

    args = parser.parse_args()
    
    model = TrainModel(
                    home=args.home,
                    save_fname=args.save_fname,
                    config_file=args.config_file,
                    fill_type=args.fill_type
                    )