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
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, accuracy_score

from data_basics import ModelBasics, get_model_metrics, create_lags
from train import TrainModel
from etl import ETL 

from itertools import islice
# import more_itertools as mit


class TestModel(ModelBasics):
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
        self.format_logs(log_type='Test', home=self.train_home)
        self.X, self.y = self.get_test_data()
        self.model = self.import_model(model_to_test)
        self.test_model(logit_clf=self.model)


    def get_test_data(self):
        """Imports testing data from ETL class

        Returns: testing dataset
        """
        logging.info(f'Testing with data from {self.test_home}.')

        Data = ETL(self.test_home, data_type='test')
        X, y = Data.split_xy(Data.test)
        return X, y


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

        logging.info(f'Loading model: {os.path.basename(model_to_load)}')
        return model


    def test_model(self, logit_clf):
        """Test the trained model on unseen data

        Returns: nothing
        """
        # print(self.X)
        y_pred = self.test_with_predictions(logit_clf, self.X).to_numpy()
        y_true = self.y.to_numpy()

        test_metrics(y_true, y_pred)
        # X = self.X.to_numpy()
        # print(accuracy_score(y_true, y_pred))
        # y = self.y.to_numpy()


        # results = get_model_metrics(logit_clf, X, y, pred_type='Test')
        # self.predicted_probabilities, self.results_msg = results


    def test_with_predictions(self, logit_clf, X, hr_lag=8):
        lag_max = hr_lag*12

        X_start = X.iloc[:lag_max]
        lag_cols=[c for c in X.columns if c.startswith('lag')]
        exog_vars = X.drop(columns=lag_cols).iloc[lag_max:]
        new_X = pd.concat([X_start, exog_vars])
        new_X.index = pd.to_datetime(new_X.index)

        ys = []
        for idx, row in new_X.iterrows():
            ts_minute = idx.minute
            curr_X = new_X.loc[new_X.index.minute == ts_minute]
            curr_row = row.to_numpy().reshape(1,-1)
            y_hat = logit_clf.predict(curr_row)

            idx_loc = new_X.index.get_loc(idx)

            for j in range(1, hr_lag + 1):
                lag_col_name = f'lag{j}_occupied'
                ind_to_set = idx_loc + 12*j

                try:
                    new_X.at[new_X.iloc[ind_to_set].name, lag_col_name] = y_hat[0]
                except:
                    continue

            ys.append((idx, y_hat[0]))
        y_hats = pd.DataFrame(ys).set_index(0)[1]
        y_hats.index.name = 'timestamp'
        return y_hats

        # print(y_hats.columns)



def test_metrics(y_true, y_hat, pred_type='Prediction Tests'):

    # y_hat = logit_clf.predict(X)
    conf_mat = pd.DataFrame(confusion_matrix(y_hat, y_true), 
                            columns = ['Vacant', 'Occupied'],
                            index = ['Vacant', 'Occupied']
                            )

    conf_mat = pd.concat([conf_mat], keys=['Actual'], axis=0)
    conf_mat = pd.concat([conf_mat], keys=['Predicted'], axis=1)

    # logging.info(f'\n{conf_mat}')
    print(f'\n{conf_mat}')

    # score = logit_clf.score(X, y)
    score = accuracy_score(y_true, y_hat)
    RMSE = np.sqrt(mean_squared_error(y_true, y_hat))

    results_msg = f'{pred_type} results on {len(y_hat)} predictions\n'\
                    f'\tScore: {score:.4}\n' \
                    f'\tRMSE: {RMSE:.4}\n'
    print(results_msg)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test models')
    parser.add_argument('-train_home', '--train_home', default='H1', type=str, help='Home to get data for, eg H1')
    parser.add_argument('-test_home', '--test_home', default=None)
    args = parser.parse_args()
    
    model = TestModel(train_home=args.train_home, test_home=args.test_home)


