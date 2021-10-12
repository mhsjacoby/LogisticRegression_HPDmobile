"""
test_model.py
Authors: Maggie Jacoby
date: 2021-06-28
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

import model_metrics as my_metrics
import prediction_functions as pred_fncs
from etl import ETL 
# from train import TrainModel



class TestModel(ETL):
    """Tests a logistic regression model.

    Reads in the pickled model and tests on unseen data.
    Can cross train and test on different homes, or same home.
    """

    def __init__(self, test_data=None, lag=8):

        self.test = test_data
        self.lag = lag
        self.X, self.y = self.split_xy(self.test)
        self.metrics = None
        self.yhat = None
        self.predictions = None

        # self.test_models(clf=model, non_prob=non_prob)


    def drop_day(self, df, name):
        drop_day = sorted(set(df.index.date))[0]
        df = df[df.index.date != drop_day]
        return df


    def test_single(self, clf):
        """Test the trained model on unseen data

        Returns: nothing
        """

        self.y = self.drop_day(self.y, name='y')
        y = self.y.to_numpy()    
        self.df = pred_fncs.test_with_predictions(logit_clf=clf, X=self.X, hr_lag=self.lag)
        self.df = self.drop_day(self.df, name='ar')
        self.yhat = self.df['Predictions'].to_numpy()
        results, pred_metrics = my_metrics.get_model_metrics(y_true=y, y_hat=self.yhat)

        # print('================ results')
        # print(results)
        # print('================ metrics')
        # print(pred_metrics)
        return pred_metrics
