"""
print_predictions.py
Authors: Maggie Jacoby
Last update: 2021-07-11
"""


import os
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, date

from etl import ETL
from train import TrainModel
from test import TestModel



local_save_path = '/Users/maggie/Desktop/Results_c01'



def test_combined(all_homes, c, metrics, coeffs):
    days = {}    
    for h in sorted(all_homes, reverse=True):
        train_homes = [x for x in all_homes if x != h]
        train_data = [all_homes[x].df for x in train_homes]
        train_df = pd.concat(train_data)
        test_data = all_homes[h]

        train_all = TrainModel(H_num=f'~{h}', train_data=train_df, cv=False, C=c)

        

        for j in range(0,len(test_data.daysets)):

            ds_df = test_data.daysets[j]
            d = [f'{d}' for d in ds_df.day.unique()]
            days[f'{h}_{j}'] = (d[0], d[-1])

            groups = list(test_data.daysets)
            test_grp = groups.pop(j)
            test = TestModel(H_num=f'~{h}', model=train_all.model, non_param=train_all.non_parametric_model, test_data=test_grp)
            test.predictions.to_csv(f'~/Desktop/predictions_test{h}_{j}_dropDay1.csv')
            test.metrics['Test'] = h
            test.metrics['group'] = j
            metrics.append(test.metrics)
    
        coeffs[f'test {h}, {c}'] = train_all.coeffs
    days = pd.DataFrame(days).transpose()
    days.columns = [['start', 'end']]
    days.to_csv('~/Desktop/day_sets.csv')
    
    return coeffs, metrics
    # return None, None


homes = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
# homes = ['H2', 'H5']

full_metrics = []
full_coeffs = {}

all_homes = {}
## Generate all datasets for all homes
for h in homes:
    x = ETL(H_num=h)
    x.generate_dataset()
    all_homes[h] = x
    print(h, 'num sets', len(x.daysets))

c = 0.1

full_coeffs, full_metrics = test_combined(all_homes=all_homes, c=c, metrics=full_metrics, coeffs=full_coeffs)




all_metrics_df = pd.concat(full_metrics)
all_metrics_df.to_csv(f'~/Desktop/all_metrics_incCO2.csv')

coeffs_df = pd.DataFrame(full_coeffs)
coeffs_df.to_csv(f'~/Desktop/all_CV_coeffs_incCO2.csv')
