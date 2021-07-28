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

    def __init__(self, H_num, hub='', train_data=None, fill_type='zeros', cv=False, C=None, penalty='l1', save_model=False, lag=8):

        super().__init__(H_num=H_num, fill_type=fill_type, lag=lag)

        # self.C = C if C else None
        self.cv = cv
        self.penalty = penalty
        self.configs = self.read_config(config_type='train', config_file='train_config')
        # self.C = self.configs['C']
        self.C = C if C else self.configs['C']

        self.train = train_data
        if self.train is None:
            super().generate_dataset(hub)
        # self.train_length = len(self.train)

        self.X, self.y = self.split_xy(self.train)
        self.model = self.train_model()
        # print('model is:')
        # print(type(self.model.coef_))
        # print(self.model.coef_)
        # print(self.model.intercept_)
        # print(self.model.get_params())

        # sys.exit()
        self.coeffs = self.format_coeffs(self.model)
        self.non_prob = self.generate_nonprobabilistic_model()


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
            if self.C:
                self.configs['C'] = self.C
            self.configs['penalty'] = self.penalty
            clf = LogisticRegression().set_params(**self.configs)


        print(f'Training model with params: {self.configs}')
        return clf


    def train_model(self, k=5):
        """Trains a logistic regression model.

        Uses default parameters, or gets params from ParameterGrid (if specified).
        Returns: sklearn logistic regression model object
        """
        print(self.X.columns)
        logit_clf = self.set_LR_parameters(k)
        X = self.X.to_numpy()
        y = self.y.to_numpy()
        # print(sum(np.isnan(X)), sum(np.isnan(y)))
        # print(np.max(X), np.max(y))
        # print(np.min(X), np.min(y))
        # print(len(X), len(y))
        logit_clf.fit(X, y)
        self.train_score = logit_clf.score(X,y)


        if self.cv:
            self.C = logit_clf.C_[0]
            # print('all Cs', logit_clf.Cs_)
        # print(logit_clf)
        # sys.exit()
        
        # self.save_model(model=logit_clf, model_name=f'ARLR_{self.H_num}')
        # sys.exit()

        coefs, intercept = self.write_new_coefs(logit_clf)
        logit_clf.coef_[0] = coefs
        logit_clf.intercept_[0] = intercept
        self.coeffs = self.format_coeffs(logit_clf)

        return logit_clf


    def save_model(self, model, model_name=''):
        """Saves model as a pickle object. 
        Names model with filename (if specified) or numerically. 
        Returns: nothing
        """
        # model_save_dir = os.path.join(self.models_dir, self.home)
        model_save_dir = '/Users/maggie/Desktop/ALRL_models'
        os.makedirs(model_save_dir, exist_ok=True)

        # if not model_name:
        #     fname = self.get_filename(model_save_dir=model_save_dir)
        # else:
        fname = f'{model_name}.pickle'
        save_name = os.path.join(model_save_dir, fname)

        if not os.path.isfile(save_name):
            pickle.dump(model, open(save_name, 'wb'))
            print(f'\t>>> Writing model to {fname}')
            # logging.info(f'Writing model to {fname}')
        else:
            pickle.dump(model, open(save_name, 'wb'))
            print(f'\t>>> Model {fname} exists. Overwriting previous')
            # logging.info(f'Overwriting previous {fname}')



    def write_new_coefs(self, model):
        coefs_file = '/Users/maggie/Documents/Github/LogisticRegression_HPDmobile/FactorialAnalysis/coeffs_10.csv'
        new_coefs = pd.read_csv(coefs_file, header=None, index_col=0)
        print('====================')
        print(new_coefs)
        print('====================')
        # return [], []
        coefs = new_coefs.values[:-1]
        coefs = np.array([x[0] for x in coefs])
        intercept = new_coefs.values[-1][0]
        return coefs, intercept


    def format_coeffs(self, model):
        # coefs, intercept = self.write_new_coefs(model)
        # model.coef_[0] = coefs
        # model.intercept_[0] = intercept
        
        coeff_df = pd.Series(model.coef_[0], index=self.X.columns)
        coeff_df = coeff_df.append(pd.Series(model.intercept_[0], index=['Intercept']))
        print(coeff_df)
        # sys.exit()
        coeff_df = coeff_df.append(pd.Series(self.C, index=['C']))
        coeff_df = coeff_df.append(pd.Series('True' if self.cv else 'False', index=['cv']))
        coeff_df = coeff_df.append(pd.Series(self.train_score, index=['accuracy']))
        coeff_df = coeff_df.append(pd.Series(self.H_num, index=['home']))
        return coeff_df


    def generate_nonprobabilistic_model(self):
        """Create likilihood of occupancy, based only on past occupancy (training data)
        """
        df = self.train.copy()

        df.insert(loc=1, column='time', value=df.index.time)
        df = df[['weekend', 'occupied', 'time']]
        model = df.groupby(['weekend', 'time']).mean()
        # model.to_csv(f'~/Desktop/NP_model_test_{self.H_num}.csv')
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
