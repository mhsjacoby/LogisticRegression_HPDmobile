"""
run_train_test.py
Authors: Maggie Jacoby
Last update: 2021-02-18
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

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix

from data_basics import ModelBasics, get_model_metrics
from etl import ETL 
from train import TrainModel
from test import TestModel


parser = argparse.ArgumentParser(description='Extract, transform, and load training/testing data')
parser.add_argument('-train_home', '--train_home', default='H1', type=str, help='Home to get data for, eg H1')
parser.add_argument('-test_home', '--test_home', default=None)
parser.add_argument('-save_fname', '--save_fname', default=None, help='Filename to save model to')
parser.add_argument('-overwrite', '--overwrite', default=False, type=bool, help='Overwrite model if it exists?')
# parser.add_argument('-print_coeffs', '--print_coeffs', default=True, type=bool, help='Print model coefficients?')
parser.add_argument('-config_file', '--config_file', default=None, help='Configuration file to use')
args = parser.parse_args()

model = TrainModel(
                home=args.train_home,
                save_fname=args.save_fname,
                overwrite=args.overwrite,
                # print_coeffs=args.print_coeffs,
                config_file=args.config_file,
                )
print(model.coeff_msg)

test_model = TestModel(train_home=args.train_home, test_home=args.test_home)

