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

from data_basics import ModelBasics, get_model_metrics, get_predictions_wGT
from train import TrainModel
from etl import ETL


class TestModel(ModelBasics):
    """Tests a logistic regression model.

    Reads in the pickled model and tests on unseen data.
    Can cross train and test on different homes, or same home.
    """

    def __init__(self, train_home, test_home, X_test=None, y_test=None, fill_type='zeros',
                model_object=None, model_name=None, save_results=True, log_flag=True):
        
        self.train_home = train_home
        self.test_home = test_home
        self.fill_type = fill_type
        self.get_directories()

        self.log_flag = log_flag
        self.format_logs(log_type='Test', home=self.train_home)

        self.yhat_df, self.gt_yhat_df = None, None
        self.predictions, self.gt_predictions = None, None
        self.results, self.gt_results = None, None
        self.conf_mat, self.gt_conf_mat = None, None

        if X_test is None or y_test is None:
            self.X, self.y = self.get_test_data()
        else:
            self.X, self.y = X_test, y_test

        if model_object is not None:
            self.model = model_object
        else:
            self.model = self.import_model(model_name)
        self.save_results = save_results

        self.test_model(self.model)


    def get_test_data(self):
        """Imports testing data from ETL class

        Returns: testing dataset
        """
        logging.info(f'Testing with data from {self.test_home}')

        Data = ETL(self.test_home, fill_type=self.fill_type, data_type='test')
        X, y = Data.split_xy(Data.test)
        return X, y


    def import_model(self, model_to_test):
        """Imports the trained model from pickle file

        Returns: sklearn logistic regression model object
        """
        if model_to_test is not None:
            model_fname = f'{self.train_home}_{model_to_test}.pickle'
        else:
            model_fname = f'{self.train_home}_*.pickle'
        possible_models = glob(os.path.join(self.models_dir, self.train_home, model_fname))

        if len(possible_models) == 0:
            print(f'\t!!! No model named {model_fname}. Exiting program.')
            sys.exit()
        model_to_load = possible_models[0]

        with open(model_to_load, 'rb') as model_file:
            model = pickle.load(model_file)  

        logging.info(f'Loading model: {os.path.basename(model_to_load)}')
        print(f'\t>>> Loading model: {os.path.basename(model_to_load)}')
        return model


    def test_model(self, logit_clf):
        """Test the trained model on unseen data

        Returns: nothing
        """
        y = self.y.to_numpy()

        self.gt_yhat_df = get_predictions_wGT(logit_clf=logit_clf, X_df=self.X)
        self.gt_predictions = self.gt_yhat_df.Predictions.to_numpy()
        self.gt_conf_mat, self.gt_results = get_model_metrics(y_true=y, y_hat=self.gt_predictions)
        logging.info(f'\n=== TESTING RESULTS USING GROUND TRUTH LAGS === \n\n{self.gt_conf_mat}')

        self.yhat_df = self.test_with_predictions(logit_clf=logit_clf, X=self.X)
        self.predictions = self.yhat_df.Predictions.to_numpy()
        self.conf_mat, self.results = get_model_metrics(y_true=y, y_hat=self.predictions)
        logging.info(f'\n=== TESTING RESULTS USING ONLY PAST PREDICTIONS === \n\n{self.conf_mat}')



    def test_with_predictions(self, logit_clf, X, hr_lag=8):
        lag_max = hr_lag*12

        X_start = X.iloc[:lag_max]
        lag_cols=[c for c in X.columns if c.startswith('lag')]
        exog_vars = X.drop(columns=lag_cols).iloc[lag_max:]
        preds_X = pd.concat([X_start, exog_vars])
        preds_X.index = pd.to_datetime(preds_X.index)

        ys = []
        for idx, row in preds_X.iterrows():
            curr_row = row.to_numpy().reshape(1,-1)
            y_hat = logit_clf.predict(curr_row)
            y_proba = logit_clf.predict_proba(curr_row)[:,1]
            idx_loc = preds_X.index.get_loc(idx)

            for j in range(1, hr_lag + 1):
                lag_col_name = f'lag{j}_occupied'
                ind_to_set = idx_loc + j*12

                try:
                    preds_X.at[preds_X.iloc[ind_to_set].name, lag_col_name] = y_hat[0]
                except:
                    continue

            ys.append((idx, y_proba[0], y_hat[0]))
        y_hats = pd.DataFrame(ys).set_index(0)
        y_hats.index.name = 'timestamp'
        y_hats.columns = ['Probability', 'Predictions']

        return y_hats


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test models')
    parser.add_argument('-train_home', '--train_home', default='H1', type=str, help='Home to get data for, eg H1')
    parser.add_argument('-test_home', '--test_home', default=None, help='Home to test on, if different from train')
    parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')

    args = parser.parse_args()
    
    test_home = args.train_home if not args.test_home else args.test_home

    Test = TestModel(
                    train_home=args.train_home,
                    test_home=test_home,
                    fill_type=args.fill_type,
                    log_flag=False
                    )

