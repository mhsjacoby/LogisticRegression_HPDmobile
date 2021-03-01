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

from data_basics import ModelBasics, get_model_metrics, get_predictions_wGT
from train import TrainModel
from etl import ETL


class TestModel(ModelBasics):
    """Tests a logistic regression model.

    Reads in the pickled model and tests on unseen data.
    Can cross train and test on different homes, or same home.
    """

    def __init__(self, H_num, train_hub, test_hub, X_test=None, y_test=None, fill_type='zeros',
                model_object=None, model_name=None, save_results=True, log_flag=True):
        self.H_num = H_num
        self.train_hub = train_hub
        self.test_hub = test_hub
        self.fill_type = fill_type
        self.get_directories()

        self.log_flag = log_flag
        self.format_logs(log_type='Test', home=self.H_num)

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
        self.results_fname = self.print_results(self.yhat_df)


    def get_test_data(self):
        """Imports testing data from ETL class

        Returns: testing dataset
        """
        logging.info(f'Testing with data from {self.test_hub}')

        Data = ETL(hub=self.test_hub, fill_type=self.fill_type, data_type='test')
        X, y = Data.split_xy(Data.test)
        return X, y


    def import_model(self, model_to_test):
        """Imports the trained model from pickle file

        Returns: sklearn logistic regression model object
        """
        if model_to_test is not None:
            model_fname = f'{self.train_hub}_{model_to_test}.pickle'
        else:
            model_fname = f'{self.train_hub}_*.pickle'
        possible_models = glob(os.path.join(self.models_dir, self.train_hub, model_fname))

        if len(possible_models) == 0:
            print(f'\t!!! No model named {model_fname}. Exiting program.')
            sys.exit()
        model_to_load = possible_models[0]

        with open(model_to_load, 'rb') as model_file:
            model = pickle.load(model_file)  

        logging.info(f'Loading model: {os.path.basename(model_to_load)}')
        print(f'\t>>> Loading model: {os.path.basename(model_to_load)}')
        return model


    def test_model(self, logit_clf, test_gt=False):
        """Test the trained model on unseen data

        Returns: nothing
        """
        # print(self.X.columns)
        # sys.exit()
        y = self.y.to_numpy()

        if test_gt:
            print('Using ground truth, NOT predictions')
            self.yhat_df =  get_predictions_wGT(logit_clf=logit_clf, X_df=self.X)
        else:
            self.yhat_df = self.test_with_predictions(logit_clf=logit_clf, X=self.X)

        self.predictions = self.yhat_df.Predictions.to_numpy()
        self.conf_mat, self.results, self.metrics = get_model_metrics(y_true=y, y_hat=self.predictions)
        # print(self.conf_mat)
        # y_hat_baseline = self.baseline_results(X=self.X)
        # # print(y_hat_baseline)
        # y_hate_baseline = y_hat_baseline.to_numpy()
        # conf_mat_bl, results_bl, blm = get_model_metrics(y_true=y, y_hat=y_hat_baseline)

        # self.metrics.update({'BL_env F1 neg': blm['F1 neg'], 'BL_env F1': blm['F1 neg'], 'BL_env Acc': blm['Accuracy']})

        self.baseline_results(X=self.X, y=y)

        # print(self.metrics)
        # sys.exit()
        logging.info(f'{pd.DataFrame(self.results)}')


    def baseline_results(self, X, y):
        """Get baseline results to compare LR to

        Uses the previous OR gate and generate predictions

        Returns: y_hat (predictions) and f1(rev) and accuracy
        """
        base_cols = ['audio', 'img']
        full_cols =  base_cols + ['temp', 'rh', 'co2eq', 'light']
 
        for cols, title in zip([base_cols, full_cols], ('AI', 'AIE')):
            df = X[cols].copy()
            pred = str('y_hat_' + title)
            df[pred] = 0 
            df.loc[df.max(axis=1) > 0.5, pred] = 1
            y_hat = df[pred].to_numpy()
            _, _, blm = get_model_metrics(y_true=y, y_hat=y_hat)
            self.metrics.update({title+' F1 neg': blm['F1 neg'], title+' F1': blm['F1'], title + ' Acc': blm['Accuracy']})

        
    def test_with_predictions(self, logit_clf, X, hr_lag=8):
        """Run data through classifier and push predictions forward as lag values

        This is used instead of get_predictions_wGT.
        Returns: probabilities (between 0,1) and predictions (0/1) as a df

        """

        lag_max = hr_lag*12

        X_start = X.iloc[:lag_max]
        lag_cols=[c for c in X.columns if c.startswith('lag')]
        exog_vars = X.drop(columns=lag_cols).iloc[lag_max:]
        preds_X = pd.concat([X_start, exog_vars])
        preds_X.index = pd.to_datetime(preds_X.index)

        ys = []

        for idx, _ in preds_X.iterrows():
            df_row = preds_X.loc[idx]
            curr_row = df_row.to_numpy().reshape(1,-1)

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
        self.additional_metrics(conf_mat)
        return fname


    def additional_metrics(self, conf_mat):

        tn, fp, fn, tp = conf_mat.ravel()

        logging.info(f'\ntn: {tn} fp:{fp} fn:{fn}, tp:{tp}')
        print(f'\ntn: {tn} fp:{fp} fn:{fn}, tp:{tp}')

        tpr = tp/(tp+fn) if tp+fn > 0 else 0.0
        fpr = fp/(tn+fp) if tn+fp > 0 else 0.0

        tnr = tn/(tn+fp) if tn+fp > 0 else 0.0
        fnr = fn/(tp+fn) if tp+fn > 0 else 0.0
        logging.info(f'tnr: {tnr:.3} fpr:{fpr:.3} fnr:{fnr:.3}, tpr:{tpr:.3}')











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

