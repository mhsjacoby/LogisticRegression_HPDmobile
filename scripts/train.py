"""
train.py
Authors: Maggie Jacoby and Jasmine Garland
Last update: 2021-04-12
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
# from model_metrics import get_model_metrics, get_predictions_wGT
import model_metrics as my_metrics
import prediction_functions as predictions
from etl import ETL 


class TrainModel(ETL):
    """Trains a logistic regression model.

    Uses ETL to load the training data.
    Writes a pickle file with the trained LR model at the end.
    """

    def __init__(self, H_num, hub='', train_data=None, fill_type='zeros', cv=False, save_model=False):

        super().__init__(H_num=H_num, fill_type=fill_type)

        self.cv = cv
        self.configs = self.read_config(config_type='train', config_file='train_config')

        self.train = train_data
        if self.train is None:
            print('> no train data passed')
            super().generate_dataset(hub)

        self.X, self.y = self.split_xy(self.train)
        self.model = self.train_model()
        self.coeffs = self.format_coeffs(self.model)

        # self.yhat_df = predictions.test_with_GT(logit_clf=self.model, X_df=self.X)



    def set_LR_parameters(self):
        """Sets the model parameters as specified in the configuration file.

        Only takes in one set of parameters.
        Returns: sklearn logistic regression model object (not fitted to data)
        """
        
        if self.cv:
            configs = {k:v for k,v in self.configs.items() if k != 'C'}
            clf = LogisticRegressionCV().set_params(**configs)

        else:
            configs = {k:v for k,v in self.configs.items() if k != 'Cs'}
            clf = LogisticRegression().set_params(**configs)

        print(f'Training model with params: {configs}')
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

        if self.cv:
            # all_cs = logit_clf.Cs_
            self.best_C = logit_clf.C_[0]
            print('best C:', self.best_C)


        # self.predictions = self.yhat_df.Predictions.to_numpy()
        # self.conf_mat, self.results, self.metrics = get_model_metrics(y_true=y, y_hat=self.predictions)
        # logging.info(f'\n=== TRAINING RESULTS === \n\n{self.conf_mat}\n')

        return logit_clf


    def format_coeffs(self, model):

        coeff_df = pd.Series(model.coef_[0], index=self.X.columns)
        coeff_df = coeff_df.append(pd.Series(model.intercept_[0], index=['Intercept']))

        return coeff_df



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-home', '--home', default='H1', type=str, help='Home to get data for, eg H1')
    parser.add_argument('-hub', '--hub', default='', type=str, help='which hub to use? (leave blank if using config file to specify)')
    parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')
    parser.add_argument('-cv', '--cv', default=False, action='store_true', help='Perform cross-validation?')
    args = parser.parse_args()

    if not args.cv:
        args.cv = False

    Data = TrainModel(
            H_num=args.home,
            hub=args.hub,
            fill_type=args.fill_type,
            cv=args.cv
            )

    # print('yhat', Data.yhat_df)
    # print('y', Data.y)

    # print('===========')
    # print(Data.coeffs)
