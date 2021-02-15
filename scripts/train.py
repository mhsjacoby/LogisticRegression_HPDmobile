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
# import argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, date

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, GridSearchCV, RepeatedKFold

from etl import ETL 
from data_basics import DataBasics


class TrainModel(DataBasics):

    def __init__(self, training_data, config_path):

        self.configs = self.read_config(config_file_path=config_path)
        self.format_logs(config_file_path=config_path, log_type='TRAIN')
        self.data = training_data
        


    def get_training_data(self):
        """
        imports training data from etl function
        """
        pass


    def split_xy(self, data):
        pass

    
    def main(self):
        print(len(self.data))
        print(self.data.columns)
        logging.info(f'length of training data: {len(self.data)}')




if __name__ == '__main__':
    data_path = '/Users/maggie/Documents/Github/LogisticRegression_HPDmobile/H1_RS4_prob.csv'
    config_json = '/Users/maggie/Documents/Github/LogisticRegression_HPDmobile/H1_config.json'
    Data = ETL(config_path=config_json, df_path=data_path)
    Model = TrainModel(config_path=config_json, training_data=Data.train)
    Model.main()