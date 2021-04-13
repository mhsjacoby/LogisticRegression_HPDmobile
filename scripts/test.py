"""
test.py
Authors: Maggie Jacoby and Jasmine Garland
Last update: 2021-02-22
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
from sklearn.metrics import confusion_matrix

# from data_basics import ModelBasics, get_model_metrics, get_predictions_wGT
# from train import TrainModel
# from etl import ETL
import model_metrics as my_metrics
import prediction_functions as pred_fncs
from etl import ETL 
from train import TrainModel



class TestModel(ETL):
    """Tests a logistic regression model.

    Reads in the pickled model and tests on unseen data.
    Can cross train and test on different homes, or same home.
    """

    # def __init__(self, H_num, train_hub, test_hub, X_test=None, y_test=None, fill_type='zeros',
    #             model_object=None, model_name=None, save_results=True, log_flag=True):
    def __init__(self, H_num, hub, model, test_data=None, fill_type='zeros', save_results=False):

        super().__init__(H_num=H_num, fill_type=fill_type)
        
        self.test = test_data
        if self.test is None:
            super().generate_dataset(hub)
        
        self.X, self.y = self.split_xy(self.test)
        # self.model = model

        # if not model:
        #     TrainModel()

        # self.yhat_df, self.gt_yhat_df = None, None
        # self.predictions, self.gt_predictions = None, None
        # self.results, self.gt_results = None, None
        # self.conf_mat, self.gt_conf_mat = None, None

        self.metrics = None
        self.yhat = None
        print('testing...')
        self.test_model(model)


        # self.save_results = save_results
        # self.results_fname = self.print_results(self.yhat_df)


    # def import_model(self, model_to_test):
    #     """Imports the trained model from pickle file

    #     Returns: sklearn logistic regression model object
    #     """
    #     if model_to_test is not None:
    #         model_fname = f'{self.train_hub}_{model_to_test}.pickle'
    #     else:
    #         model_fname = f'{self.train_hub}_*.pickle'
    #     possible_models = glob(os.path.join(self.models_dir, self.train_hub, model_fname))

    #     if len(possible_models) == 0:
    #         print(f'\t!!! No model named {model_fname}. Exiting program.')
    #         sys.exit()
    #     model_to_load = possible_models[0]

    #     with open(model_to_load, 'rb') as model_file:
    #         model = pickle.load(model_file)  

    #     logging.info(f'Loading model: {os.path.basename(model_to_load)}')
    #     print(f'\t>>> Loading model: {os.path.basename(model_to_load)}')
    #     return model


    def test_model(self, clf, test_gt=False):
        """Test the trained model on unseen data

        Returns: nothing
        """
        y = self.y.to_numpy()
        metrics = {}
        metrics = pred_fncs.baseline_OR(X=self.X, y=y, metrics=metrics)

        gt_df = pred_fncs.test_with_GT(logit_clf=clf, X_df=self.X)
        gt_yhat = gt_df['Predictions'].to_numpy()
        _, gt_metrics = my_metrics.get_model_metrics(y_true=y, y_hat=gt_yhat)
        metrics['ground truth'] = gt_metrics

        self.df = pred_fncs.test_with_predictions(logit_clf=clf, X=self.X)
        self.yhat = self.df['Predictions'].to_numpy()
        _, pred_metrics = my_metrics.get_model_metrics(y_true=y, y_hat=self.yhat)
        metrics['predictions'] = pred_metrics

        self.metrics = pd.DataFrame(metrics).transpose()
        




    def print_results(self, y_hat, i=0):
        y_hat['Occupied'] = self.y
        os.makedirs(self.results_csvs, exist_ok=True)

        results_name = f'{self.train_hub}_{self.test_hub}_{self.fill_type}_*.csv'
        existing_results = glob(os.path.join(self.results_csvs, self.H_num, results_name))
        if len(existing_results) > 0:
            i = int(sorted(existing_results)[-1].split('_')[-1].strip('.csv'))
        fname = f'{self.train_hub}_{self.test_hub}_{self.fill_type}_{str(i+1)}.csv'

        os.makedirs(os.path.join(self.results_csvs, self.H_num), exist_ok=True)
        y_hat.to_csv(os.path.join(self.results_csvs, self.H_num, fname), index_label='timestamp')
        logging.info(f'\nSaving results to {fname}')

        conf_mat = confusion_matrix(y_hat['Occupied'], y_hat['Predictions']).ravel()
        self.conf_results = my_metrics.counts(conf_mat)
        return fname








# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Test models')
#     parser.add_argument('-train_home', '--train_home', default='H1', type=str, help='Home to get data for, eg H1')
#     parser.add_argument('-test_home', '--test_home', default=None, help='Home to test on, if different from train')
#     parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')

#     args = parser.parse_args()
    
#     test_home = args.train_home if not args.test_home else args.test_home

#     Test = TestModel(
#                     train_home=args.train_home,
#                     test_home=test_home,
#                     fill_type=args.fill_type,
#                     log_flag=False
#                     )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-home', '--home', default='H1', type=str, help='Home to get data for, eg H1')
    parser.add_argument('-hub', '--hub', default='', type=str, help='which hub to use? (leave blank if using config file to specify)')
    parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')
    parser.add_argument('-cv', '--cv', default=False, action='store_true', help='Perform cross-validation?')
    args = parser.parse_args()

    if not args.cv:
        args.cv = False
    
    Data = ETL(
            H_num=args.home,
            # hub=args.hub,
            fill_type=args.fill_type
            )
    Data.generate_dataset(hub=args.hub)

    model = TrainModel(
            H_num=args.home,
            hub=args.hub,
            fill_type=args.fill_type,
            cv=args.cv,
            train_data=Data.train
            )

    test = TestModel(
            H_num=args.home,
            hub=args.hub,
            fill_type=args.fill_type,
            model=model.model,
            test_data=Data.test
            )
    print(test.metrics  )