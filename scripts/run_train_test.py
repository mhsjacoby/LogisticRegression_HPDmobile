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



local_save_path = '/Users/maggie/Desktop/excel_results'


def run_train_test(train_hub, test_hub, h='H1', fill_type='zeros'):

    print(f'\n\t======== Running train/test with hubs {train_hub}/{test_hub} ========\n')

    print('ETL...')
    Data = ETL(
            H_num=h,
            hub=train_hub,
            data_type='train and test',
            fill_type=fill_type,
            log_flag=False
            )
    X_train, y_train = Data.split_xy(Data.train)


    print('Training...')
    Model = TrainModel(
                    H_num=h,
                    hub=train_hub,
                    X_train=X_train,
                    y_train=y_train,
                    fill_type=fill_type,
                    )

    if test_hub == train_hub:
        X_test, y_test = Data.split_xy(Data.test)
    else:
        testData = ETL(
            H_num=h,
            hub=test_hub,
            data_type='test',
            fill_type=fill_type
            )
        X_test, y_test = testData.split_xy(testData.test)

    
    print('Testing...')
    Test_model = TestModel(
                    H_num=h,
                    train_hub=train_hub,
                    test_hub=test_hub,
                    X_test=X_test,
                    y_test=y_test,
                    model_object=Model.model,
                    fill_type=fill_type,
                    )

    return Test_model.metrics, Test_model.results_fname, Model.coeff_df, Model.best_C



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Join ETL, Train, and Test functionality.')
    parser.add_argument('-train_home', '--train_home', default='H1', type=str, help='Home to train on, eg H1')
    parser.add_argument('-test_home', '--test_home', default=None, help='Home to test on, if different from train')
    parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')
    parser.add_argument('-suffix', '--suffix', default='', type=str, help='Descriptive suffix to append to results files.')
    parser.add_argument('-hubs', '--hubs', default='', nargs='+', help='Hubs to use.')
    parser.add_argument('-system', '--system', default='R', type=str, help='Sysem color (R or B)')
    args = parser.parse_args()

    test_home = args.train_home if not args.test_home else args.test_home
    train_home = args.train_home
    curr_fill = args.fill_type
    suffix = args.suffix
    color = args.system
    hubs_used = [1,2,3,4,5] if not args.hubs else args.hubs

    list_of_hubs = [f'{train_home}{color}S{str(i)}' for i in hubs_used]

    hub_results, coeff_list, coeff_cols = [], [], []

    for pair in itertools.product(list_of_hubs, repeat=2):
        h_num = train_home
        tr, ts = pair
        
        run, fname, coeffs, best_c = run_train_test(
                                                    h=h_num,
                                                    train_hub=tr,
                                                    test_hub=ts, 
                                                    fill_type=curr_fill
                                                    )
        
        run_df = pd.DataFrame(data=run, index=[0])

        self_cross = 'self' if tr == ts else 'cross'
        if tr == ts:
            coeff_list.append(coeffs)
            coeff_cols.append(tr.replace(f'{h_num}RS', 'hub-'))

        run_df[['Train', 'Test', 'C', 'Fill', 'Self/Cross', 'fname']] = [tr, ts, best_c, curr_fill, self_cross, fname]
        hub_results.append(run_df)


    df = pd.concat(hub_results, axis=0)
    df.to_csv(os.path.join(local_save_path, f'{h_num}_{curr_fill}_{suffix}.csv'), index=False)
    coeffs_df = pd.concat(coeff_list, axis=1)
    coeffs_df.columns=coeff_cols
    coeffs_df.to_csv(os.path.join(local_save_path, f'{h_num}_{curr_fill}_coeffs_{suffix}.csv'), index=True)
