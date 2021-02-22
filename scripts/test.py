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

from data_basics import ModelBasics, get_model_metrics
from train import TrainModel
from etl import ETL


class TestModel(ModelBasics):
    """Tests a logistic regression model.

    Reads in the pickled model and tests on unseen data.
    Can cross train and test on different homes, or same home.
    """

    def __init__(self, train_home, test_home, X_test=None, y_test=None, fill_type='zeros',
                model_object=None, model_name=None, save_results=True):
        
        self.train_home = train_home
        self.test_home = test_home
        self.fill_type = fill_type
        self.get_directories()
        self.test_log = self.format_logs(log_type='Test', home=self.train_home)

        if X_test is None or y_test is None:
            self.X, self.y = self.get_test_data()
        else:
            self.X, self.y = X_test, y_test

        if model_object is not None:
            self.model = model_object
        else:
            self.model = self.import_model(model_name)
        self.save_results = save_results


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
        # print(self.X)
        counts = get_counts(df=self.X, stage='test', counts=self.counts)

        # df = self.X
        # print(f'Fill type {self.fill_type} Testing')
        # print(f'images 0s {len(df[df.img == 0])}')
        # print(f'images 1s {len(df[df.img == 1])}')
        # print(f'total length {len(df)}')

        y_pred = self.test_with_predictions(logit_clf, self.X).to_numpy()
        y_true = self.y.to_numpy()

        test_metrics(y_true, y_pred)
        # print('Testing with ground truth')

        # X = self.X.to_numpy()
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

        print(y_hats.columns)



# def test_metrics(y_true, y_hat, pred_type='Prediction Tests'):

#     # y_hat = logit_clf.predict(X)
#     conf_mat = pd.DataFrame(confusion_matrix(y_hat, y_true), 
#                             columns = ['Vacant', 'Occupied'],
#                             index = ['Vacant', 'Occupied']
#                             )

#     conf_mat = pd.concat([conf_mat], keys=['Actual'], axis=0)
#     conf_mat = pd.concat([conf_mat], keys=['Predicted'], axis=1)

#     # logging.info(f'\n{conf_mat}')
#     print(f'\n{conf_mat}')

#     # score = logit_clf.score(X, y)
#     score = accuracy_score(y_true, y_hat)
#     RMSE = np.sqrt(mean_squared_error(y_true, y_hat))

#     results_msg = f'{pred_type} results on {len(y_hat)} predictions\n'\
#                     f'\tScore: {score:.4}\n' \
#                     f'\tRMSE: {RMSE:.4}\n'
#     print(results_msg)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test models')
    parser.add_argument('-train_home', '--train_home', default='H1', type=str, help='Home to get data for, eg H1')
    parser.add_argument('-test_home', '--test_home', default=None, help='Home to test on, if different from train')
    parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')

    args = parser.parse_args()
    
    test_home = args.train_home if not args.test_home else args.test_home

    model = TestModel(
                    train_home=args.train_home,
                    test_home=test_home,
                    )


