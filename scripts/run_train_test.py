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



# x_ones = []
# x_zeros = []
# df_lens = []

# keep_count = (x_ones, x_zeros, df_lens)








parser = argparse.ArgumentParser(description='Extract, transform, and load training/testing data')
parser.add_argument('-train_home', '--train_home', default='H1', type=str, help='Home to get data for, eg H1')
parser.add_argument('-test_home', '--test_home', default=None)
parser.add_argument('-save_fname', '--save_fname', default=None, help='Filename to save model to')
parser.add_argument('-overwrite', '--overwrite', default=False, type=bool, help='Overwrite model if it exists?')
parser.add_argument('-print_train', '--print_train', default=True, type=bool, help='Print model coefficients?')
parser.add_argument('-config_file', '--config_file', default=None, help='Configuration file to use')
parser.add_argument('-fill_type', '--fill_type')
args = parser.parse_args()



test_home = args.train_home if not args.test_home else args.test_home
counts = ([], [], [])

Data = ETL(
        home=args.train_home,
        data_type='train and test',
        fill_type=args.fill_type,
        counts=counts
        )


Model = TrainModel(
                home=args.train_home,
                save_fname=args.save_fname,
                overwrite=args.overwrite,
                config_file=args.config_file,
                fill_type=args.fill_type,
                counts=Data.counts
                )

# if args.print_train:
    # print(model.coeff_msg)
    # print(model.results_msg)

Test_model = TestModel(
                train_home=args.train_home,
                test_home=args.test_home,
                model_to_test=args.save_fname,
                fill_type=args.fill_type,
                counts=Model.counts)
print('\n==================================')
print(f'Model trained/tested on {args.train_home}/{test_home} with fill type {args.fill_type} using predictions')
# print(test_model.results_msg)

# print(test_model.predicted_probabilities)
print(Test_model.counts)
