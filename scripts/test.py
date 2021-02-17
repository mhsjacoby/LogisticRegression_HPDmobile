"""
test.py
Authors: Maggie Jacoby and Jasmine Garland
Last update: 2021-02-16
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
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix


from data_basics import DataBasics
from train import TrainModel
from etl import ETL 


class TestModel(DataBasics):
"""Tests a logistic regression model.

Reads in the pickled model and tests on unseen data.
Can cross train and test on different homes, or same home.
"""

    def __init__(self, train_home, test_home=None, model_to_test=None):

        self.train_home = train_home
        if not test_home:
            self.test_home = train_home
        else:
            self.test_home = test_home

        self.get_directories()
        self.configs = None

        test_data = self.get_test_data()
        self.X, self.y = self.split_xy(test_data)
        self.model = self.import_model(model_to_test)
        self.test_model(logit_clf=self.model)



    def get_test_data(self):
        """Imports testing data from ETL class

        Returns: testing dataset
        """
        self.format_logs(log_type='Test', home=self.train_home)

        Data = ETL(self.test_home, data_type='test')
        return Data.test


    def import_model(self, model_to_test):
        """Imports the trained model from pickle file

        Returns: sklearn logistic regression model object
        """
        if not model_to_test:
            model_fname = '*_model.pickle'
        else:
            model_fname = f'{self.train_home}_{model_to_test}.pickle'
        possible_models = glob(os.path.join(self.models_dir, self.train_home, model_fname))

        if len(possible_models) == 0:
            print(f'No model named {model_name}. Exiting program.')
            sys.exit()
        elif len(possible_models) > 1:
            print(f'{len(model_to_load)} possible models. Exiting program.')
            sys.exit()
        else:
            model_to_load = possible_models[0]

        with open(model_to_load, 'rb') as model_file:
            model = pickle.load(model_file)  
        return model


    def test_model(self, logit_clf):
        """Test the trained model on unseen data

        Returns: nothing
        """
        self.X = self.X.drop(columns = ['day'])
        X = self.X.to_numpy()
        y = self.y.to_numpy()

        probs = logit_clf.predict(X)
        y_hat = pd.Series(probs).apply(lambda x: 1 if (x > 0.5) else 0)

        conf_mat = pd.DataFrame(confusion_matrix(y_hat, y), 
                            columns = ['Unoccupied', 'Occupied'], index = ['Unoccupied', 'Occupied'])
        logging.info(f'\n{conf_mat}\n')
        logging.info(f'Test score: {logit_clf.score(X, y):.4} on {len(X)} predictions')
        logging.info(f'RMSE: {np.sqrt(mean_squared_error(y, y_hat))}')

        print(f'RMSE: {np.sqrt(mean_squared_error(y, y_hat))}')
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract, transform, and load training/testing data')
    parser.add_argument('-train_home', '--train_home', default='H1', type=str, help='Home to get data for, eg H1')
    args = parser.parse_args()
    
    model = TestModel(train_home=args.train_home)


