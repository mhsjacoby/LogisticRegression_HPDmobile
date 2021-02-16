"""
train_LRmodel.py
Author: Maggie Jacoby
Base code provided by Jasmine Garland
February, 2021

Description
Inputs
Outputs
"""

import os
import sys
import csv
import logging
# import json
import argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, date

from sklearn.linear_model import LogisticRegression
# from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
# from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import ParameterGrid #KFold, GridSearchCV, RepeatedKFold

from etl import ETL 
from data_basics import DataBasics


class TrainModel(DataBasics):

    def __init__(self, home):

        self.home = home
        self.get_directories()

        config_files = glob(os.path.join(self.config_dir, f'{self.home}_train_*'))
        self.configs = self.read_config(config_files=config_files)

        train_data, test_data = self.get_data()
        self.X, self.y = self.split_xy(train_data)

        self.test_X, self.test_y = self.split_xy(test_data)
        self.train_model()


    def get_data(self, data_type='train and test'):
        """
        imports training data from etl function
        """
        
        Data = ETL(self.home, data_type=data_type)

        return Data.train, Data.test


    def split_xy(self, df):

        y = df['occupied']

        X = df[df.columns.difference(['occupied'], sort=False)]
        # print(X.columns)
        
        return X, y


    def train_model(self):
        # logit_clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.02)
        parameter_grid = ParameterGrid(self.configs)
        # sys.exit()
        
        # logit_clf = LogisticRegression()
        for param in parameter_grid:
            logit_clf = LogisticRegression(**param)
        # for k, v in self.configs.items():
            # logit_clf = LogisticRegression().set_params(**{k: v})
            print(logit_clf)
        self.X = self.X.drop(columns = ['day'])
        X = self.X.to_numpy()
        y = self.y.to_numpy()
        logit_clf.fit(X, y)
        self.test_X = self.test_X.drop(columns = ['day'])
        test_X = self.test_X.to_numpy()
        test_y = self.test_y.to_numpy()
        probs = logit_clf.predict(test_X)
        print('probs', probs)
        preds = pd.Series(probs).apply(lambda x: 1 if (x > 0.5) else 0)

        mat = pd.DataFrame(confusion_matrix(preds, test_y), 
                            columns = ["Unoccupied", "Occupied"], index = ["Unoccupied", "Occupied"])
        
        r_squared = r2_score(test_y, preds)
        print("R-Squared:", r2_score(test_y, preds))
        print("RMSE:", np.sqrt(mean_squared_error(test_y, preds)))
        print("test score:", logit_clf.score(test_X, test_y))
        print("train score:", logit_clf.score(X, y))
        print(pd.Series(logit_clf.coef_[0], index = self.X.columns))
        print('intercept:', logit_clf.intercept_)
        # print(logit_clf.coef_)
        # print(len(self.X.columns))






    
    def main(self):
        print(len(self.data))
        print(self.data.columns)
        logging.info(f'length of training data: {len(self.data)}')




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract, transform, and load training/testing data')
    parser.add_argument('-home', '--home', default='H1', type=str, help='Home to get data for, eg H1')
    args = parser.parse_args()
    
    home = args.home

    model = TrainModel(home)