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
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix

from data_basics import DataBasics
from etl import ETL 


class TrainModel(DataBasics):
    """Trains a logistic regression model.

    Uses ETL to load the training data.
    Writes a pickle file with the trained LR model at the end.
    """

    def __init__(self, home, save_fname=None, overwrite=False, print_coeffs=False, config_file=None):

        self.home = home
        self.get_directories()
        self.configs = None

        train_data = self.get_train_data(config_file)
        self.X, self.y = self.split_xy(train_data)
        self.model = self.train_model(print_coeffs=print_coeffs)
        self.save_model(model=self.model, model_name=save_fname, overwrite=overwrite)

    def get_train_data(self, config_file=None):
        """Imports training data from ETL class

        Returns: training dataset
        """
        self.format_logs(log_type='Train', home=self.home)

        if not config_file:
            config_files = glob(os.path.join(self.config_dir, f'{self.home}_train_*.yaml'))
        else:
            print(f'Configuration file specified: {config_file}.')
            config_files = glob(os.path.join(self.config_dir, config_file))

        self.configs = self.read_config(config_files=config_files, config_type='Train')

        logging.info('Parameters used:')
        for param in self.configs:
            logging.info(f'\t{param}: {self.configs[param]}')        

        Data = ETL(self.home, data_type='train')
        return Data.train

    def set_LR_parameters(self):
        """Sets the model parameters as specified in the configuration file.

        Only takes in one set of parameters.
        Returns: sklearn logistic regression model object (not fit)
        """
        clf = LogisticRegression().set_params(**self.configs)
        return clf

    def train_model(self, print_coeffs):
        """Trains a logistic regression model.

        Uses default parameters, or gets params from ParameterGrid (if specified).
        Returns: sklearn logistic regression model object
        """
        logit_clf = self.set_LR_parameters()
        # logit_clf = LogisticRegression(solver='saga', penalty='l1', max_iter=1000, C=.02)

        self.X = self.X.drop(columns = ['day'])
        X = self.X.to_numpy()
        y = self.y.to_numpy()
        logit_clf.fit(X, y)

        coeff_msg = f'{pd.Series(logit_clf.coef_[0], index = self.X.columns).to_string()}\n' \
                        f'intercept\t{logit_clf.intercept_[0]}'
        logging.info(f'\nCoefficients\n{coeff_msg}\n')
        logging.info(f'Train score: {logit_clf.score(X, y):.4} on {len(X)} predictions')

        if print_coeffs:
            print(coeff_msg)
            print(f'train score: {logit_clf.score(X, y)}')
        return logit_clf
        
    def get_filename(self, model_save_dir):
        """Gets model number if filename not specified.

        Increments name based on total number of models.
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
            print(f'Saving model to {save_name}') 
        else:
            if overwrite:
                print(f'Model {fname} exists. Overwriting previous model with current one.')
                pickle.dump(model, open(save_name, 'wb'))
                logging.info(f'Overwriting previous {fname}.')
            else:
                print(f'\tModel {fname} already exists.\n' \
                        '\tPlease re-run and specify a filename or set overwrite to "True".\n' \
                        '\tProgram exiting without saving model.')
                logging.info('No model was written.')
                sys.exit()
        logging.info(f'Saving model to: {os.path.relpath(save_name, start=self.models_dir)}')

    def main(self):
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract, transform, and load training/testing data')
    parser.add_argument('-home', '--home', default='H1', type=str, help='Home to get data for, eg H1')
    parser.add_argument('-save_fname', '--save_fname', default=None, help='Filename to save model to')
    parser.add_argument('-overwrite', '--overwrite', default=False, type=bool, help='Overwrite model if it exists?')
    parser.add_argument('-print_coeffs', '--print_coeffs', default=False, type=bool, help='Print model coefficients?')
    parser.add_argument('-config_file', '--config_file', default=None, help='Configuration file to use')
    args = parser.parse_args()
    
    model = TrainModel(
                    home=args.home,
                    save_fname=args.save_fname,
                    overwrite=args.overwrite,
                    print_coeffs=args.print_coeffs,
                    config_file=args.config_file,
                    )