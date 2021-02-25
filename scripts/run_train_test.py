"""
run_train_test.py
Authors: Maggie Jacoby
Last update: 2021-02-22
"""

import os
import sys
import argparse
from glob import glob
import numpy as np
import pandas as pd
import itertools

from train import TrainModel
from test import TestModel
from etl import ETL


parser = argparse.ArgumentParser(description='Join ETL, Train, and Test functionality.')
parser.add_argument('-train_home', '--train_home', default='H1', type=str, help='Home to train on, eg H1')
parser.add_argument('-test_home', '--test_home', default=None, help='Home to test on, if different from train')
parser.add_argument('-hub', '--hub', default='RS4', type=str, help='Which hub to use')

parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')
args = parser.parse_args()

test_home = args.train_home if not args.test_home else args.test_home



def run_train_test(train_home, test_home, fill_type='zeros'):

    print(f'\n\t======== Running train/test with hubs {train_home}/{test_home} ========\n')

    print('ETL...')
    
    Data = ETL(
            home=train_home,
            data_type='train and test',
            fill_type=fill_type,
            log_flag=False
            )
    X_train, y_train = Data.split_xy(Data.train)


    print('Training...')
    Model = TrainModel(
                    home=train_home,
                    X_train=X_train,
                    y_train=y_train,
                    fill_type=fill_type,
                    )
    # print(Model.coeff_msg)


    if test_home == train_home:
        X_test, y_test = Data.split_xy(Data.test)
    else:
        testData = ETL(
            home=test_home,
            data_type='test',
            fill_type=fill_type
            )
        X_test, y_test = testData.split_xy(testData.test)

    
    print('Testing...')
    Test_model = TestModel(
                    train_home=train_home,
                    test_home=test_home,
                    X_test=X_test,
                    y_test=y_test,
                    model_object=Model.model,
                    fill_type=fill_type,
                    )
    # print(Model.coeff_df)
    return Test_model.metrics, Test_model.results_fname, Model.coeff_df


curr_fill = args.fill_type
list_of_hubs = ['H1RS1', 'H1RS2', 'H1RS3', 'H1RS4', 'H1RS5']
hub_results, coeff_list, coeff_cols = [], [], []

for pair in itertools.product(list_of_hubs, repeat=2):
    tr, ts = pair
    run, fname, coeffs = run_train_test(train_home=tr, test_home=ts, fill_type=curr_fill)
    run_df = pd.DataFrame(data=run, index=[0])
    self_cross = 'self' if tr == ts else 'cross'
    if tr == ts:
        coeff_list.append(coeffs)
        coeff_cols.append(tr.replace('H1RS', 'hub-'))
    else:
        continue
#     run_df[['Train', 'Test', 'Fill', 'Self/Cross', 'fname']] = [
#             tr.replace('H1RS', 'hub-'), 
#             ts.replace('H1RS', 'hub-'), 
#             args.fill_type,
#             self_cross,
#             fname
#             ]
#     hub_results.append(run_df)
#     sys.exit()
# df = pd.concat(hub_results, axis=0)
# print(df)
# df.to_csv(os.path.join('/Users/maggie/Desktop/excel_results', f'H1_{curr_fill}.csv'), index=False)
coeffs_df = pd.concat(coeff_list, axis=1)
coeffs_df.columns=coeff_cols
print(coeffs_df)
coeffs_df.to_csv(os.path.join('/Users/maggie/Desktop/excel_results', f'H1_{curr_fill}_coeffs.csv'), index=True)

# def parse_coeff_msg()