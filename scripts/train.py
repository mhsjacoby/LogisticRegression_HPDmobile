"""
train.py
Authors: Maggie Jacoby and Jasmine Garland
Last update: 2021-04-15
"""

import os
import sys
import csv
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, date
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit

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
        self.C = self.configs['C']

        self.train = train_data
        if self.train is None:
            super().generate_dataset(hub)

        self.X, self.y = self.split_xy(self.train)
        self.model = self.train_model()
        self.coeffs = self.format_coeffs(self.model)
        self.non_parametric_model = self.generate_nonparametric_model()


    def set_LR_parameters(self, k):
        """Sets the model parameters as specified in the configuration file.

        Only takes in one set of parameters.
        Returns: sklearn logistic regression model object (not fitted to data)
        """
        
        if self.cv:
            tscv = TimeSeriesSplit(n_splits=k)
            self.configs = {k:v for k,v in self.configs.items() if k != 'C'}
            self.configs['cv'] = tscv
            clf = LogisticRegressionCV().set_params(**self.configs)
            
        else:
            self.configs = {k:v for k,v in self.configs.items() if k != 'Cs' and k != 'cv'}
            clf = LogisticRegression().set_params(**self.configs)

        print(f'Training model with params: {self.configs}')
        return clf


    def train_model(self, k=5):
        """Trains a logistic regression model.

        Uses default parameters, or gets params from ParameterGrid (if specified).
        Returns: sklearn logistic regression model object
        """
        logit_clf = self.set_LR_parameters(k)
        X = self.X.to_numpy()
        y = self.y.to_numpy()
        logit_clf.fit(X, y)

        if self.cv:
            self.C = logit_clf.C_[0]
            print('all Cs', logit_clf.Cs_)
        return logit_clf


    def format_coeffs(self, model):

        coeff_df = pd.Series(model.coef_[0], index=self.X.columns)
        coeff_df = coeff_df.append(pd.Series(model.intercept_[0], index=['Intercept']))
        coeff_df = coeff_df.append(pd.Series(self.C, index=['C']))
        coeff_df = coeff_df.append(pd.Series('True' if self.cv else 'False', index=['cv']))

        return coeff_df


    def generate_nonparametric_model(self):
        """Create likilihood of occupancy, based only on past occupancy (training data)
        """
        df = self.train.copy()

        df.insert(loc=1, column='time', value=df.index.time)
        df = df[['weekend', 'occupied', 'time']]
        model = df.groupby(['weekend', 'time']).mean()
        return model



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
