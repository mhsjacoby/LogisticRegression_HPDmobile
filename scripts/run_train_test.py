"""
run_train_test.py
Authors: Maggie Jacoby
Last update: 2021-02-18
"""

import os
import sys
import argparse
from glob import glob

from train import TrainModel
from test import TestModel
from etl import ETL


parser = argparse.ArgumentParser(description='Join ETL, Train, and Test functionality.')
parser.add_argument('-train_home', '--train_home', default='H1', type=str, help='Home to train on, eg H1')
parser.add_argument('-test_home', '--test_home', default=None, help='Home to test on, if different from train')
parser.add_argument('-save_fname', '--save_fname', default=None, help='Filename to save model to')
parser.add_argument('-print_train', '--print_train', default=False, type=bool, help='Print model coefficients?')
parser.add_argument('-config_file', '--config_file', default=None, help='Configuration file to use')
parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')
args = parser.parse_args()

test_home = args.train_home if not args.test_home else args.test_home

print('ETL...')
Data = ETL(
        home=args.train_home,
        data_type='train and test',
        fill_type=args.fill_type,
        )

X_train, y_train = Data.split_xy(Data.train)
print('Training...')
Model = TrainModel(
                home=args.train_home,
                X_train=X_train,
                y_train=y_train,
                fill_type=args.fill_type,
                )

X_test, y_test = Data.split_xy(Data.test)
print('Testing...')
Test_model = TestModel(
                train_home=args.train_home,
                test_home=test_home,
                X_test=X_test,
                y_test=y_test,
                model_object=Model.model,
                fill_type=args.fill_type,
                )


# print('\n==================================')
# print(f'Model trained/tested on {args.train_home}/{test_home} with fill type {args.fill_type} using predictions')
# # print(test_model.results_msg)

# # print(test_model.predicted_probabilities)
# # print(Test_model.counts)
