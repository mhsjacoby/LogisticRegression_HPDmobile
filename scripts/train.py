"""
train.py
Authors: Maggie Jacoby and Jasmine Garland
Last update: 2021-02-24
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

from data_basics import ModelBasics, get_model_metrics, get_predictions_wGT
from etl import ETL 


class TrainModel(ModelBasics):
    """Trains a logistic regression model.

    Uses ETL to load the training data.
    Writes a pickle file with the trained LR model at the end.
    """

    def __init__(self, H_num, hub, X_train=None, y_train=None, fill_type='zeros',
                save_model=False, save_fname=None, config_file=None, log_flag=True):

        self.H_num = H_num
        self.hub = hub
        self.fill_type = fill_type
        self.get_directories()

        self.coeff_msg = None
        self.yhat_df = None
        self.predictions = None
        self.results = None
        self.conf_mat = None

        self.log_flag = log_flag
        self.format_logs(log_type='Train', home=self.H_num)
        config_file_list = self.pick_config_file(config_file)
        self.configs = self.read_config(config_files=config_file_list, config_type='Train')

        if X_train is None or y_train is None:
            self.X, self.y = self.get_train_data()
        else:
            self.X, self.y = X_train, y_train

        self.model = self.train_model()
        if save_model:
            self.save_model(model=self.model, model_name=save_fname)



    def pick_config_file(self, config_file=None):
        """
        """
        if not config_file:
             config_file_list = glob(os.path.join(self.config_dir, f'H1_train_*.yaml'))
            # config_file_list = glob(os.path.join(self.config_dir, f'{self.H_num}_train_*.yaml'))
            # config_file_list = glob(os.path.join(self.config_dir, f'{self.home}_train_*.yaml'))
        else:
            config_file_list = glob(os.path.join(self.config_dir, config_file))
        return config_file_list


    def get_train_data(self):
        """Imports training data from ETL class

        Returns: training dataset
        """     

        Data = ETL(H_num=self.H_num, hub=self.hub, fill_type=self.fill_type, data_type='train')
        X, y = Data.split_xy(Data.train)
        return X, y


    def set_LR_parameters(self):
        """Sets the model parameters as specified in the configuration file.

        Only takes in one set of parameters.
        Returns: sklearn logistic regression model object (not fitted to data)
        """
        logging.info('Parameters used:')
        for param in self.configs:
            logging.info(f'\t{param}: {self.configs[param]}')
        
        print(f'\t>>> Training model with params: {self.configs}')

        # clf = LogisticRegression().set_params(**self.configs)

        clf = LogisticRegressionCV().set_params(**self.configs)
        # print(clf.Cs_)
        # sys.exit()
        return clf


    def train_model(self):
        """Trains a logistic regression model.

        Uses default parameters, or gets params from ParameterGrid (if specified).
        Returns: sklearn logistic regression model object
        """
        logit_clf = self.set_LR_parameters()
        X = self.X.to_numpy()
        y = self.y.to_numpy()
        logit_clf.fit(X, y)
        # print(logit_clf.Cs_)
        # sys.exit()
        self.best_C = logit_clf.C_[0]
        print(self.best_C)


        self.coeff_msg = self.print_coeffs(logit_clf)
        logging.info(f'{self.coeff_msg}')

        self.yhat_df = get_predictions_wGT(logit_clf=logit_clf, X_df=self.X)
        self.predictions = self.yhat_df.Predictions.to_numpy()
        self.conf_mat, self.results, self.metrics = get_model_metrics(y_true=y, y_hat=self.predictions)
        # logging.info(f'\n=== TRAINING RESULTS === \n\n{self.conf_mat}\n')

        return logit_clf


    def print_coeffs(self, model):

        coeff_msg = f'\nCoefficients:\n{pd.Series(model.coef_[0], index=self.X.columns).to_string()}\n' \
                        f'intercept\t{model.intercept_[0]}'
        self.coeff_df = pd.Series(model.coef_[0], index=self.X.columns)
        self.coeff_df = self.coeff_df.append(pd.Series(model.intercept_[0], index=['Intercept']))
        return coeff_msg


    def get_filename(self, model_save_dir):
        """Gets model number if filename not specified.

        Increments name based on total number of models.
        Params:

        Returns: filename to save model as 
        """
        models = glob(os.path.join(model_save_dir, '*.pickle'))
        if not models:
            fname = f'{self.hub}_model.pickle'
        else:
            model_num = len(models) + 1
            fname = f'{self.hub}_model_{model_num}.pickle'
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
            print(f'\t>>> Writing model to {fname}')
            logging.info(f'Writing model to {fname}')
        else:
            pickle.dump(model, open(save_name, 'wb'))
            print(f'\t>>> Model {fname} exists. Overwriting previous')
            logging.info(f'Overwriting previous {fname}')


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Train and save models')
#     parser.add_argument('-home', '--home', default='H1', type=str, help='Home to get data for, eg H1')
#     parser.add_argument('-save_fname', '--save_fname', default=None, help='Filename to save model to')
#     parser.add_argument('-config_file', '--config_file', default=None, help='Configuration file to use')
#     parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')

#     args = parser.parse_args()
    
#     Model = TrainModel(
#                     home=args.home,
#                     config_file=args.config_file,
#                     # config_file='/Users/maggie/Documents/Github/LogisticRegression_HPDmobile/configuration_files/H1_train_config.yaml',
#                     fill_type=args.fill_type,
#                     save_model=True,
#                     save_fname=args.save_fname,
#                     log_flag=False
#                     )

