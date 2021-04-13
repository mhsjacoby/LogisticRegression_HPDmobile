"""
train_test2.py
Authors: Maggie Jacoby
Last update: 2021-04-13
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
from train import TrainModel
from test import TestModel

local_save_path = '/Users/maggie/Desktop'


H1 = TrainModel(
        H_num='H1',
        hub=''
        )

H2 = TrainModel(
        H_num='H2',
        hub=''
        )

homes = [H1, H2]

all_metrics, coeff_list = [], {}

for train in homes:
    coeff_list[train.H_num] = train.coeffs

    for test in homes:
        print(f'Model trained on {train.H_num}, tested on data from {test.H_num}')
        
        T = TestModel(
            H_num=test.H_num,
            model=train.model,
            test_data=test.test
            )

        metric_df = T.metrics
        metric_df.index.name = 'function'
        metric_df['train'] = train.H_num
        metric_df['test'] = test.H_num
        metric_df.set_index(['train', 'test', metric_df.index], inplace=True)
        all_metrics.append(metric_df)


df = pd.concat(all_metrics)
print(df)
df.to_csv(os.path.join(local_save_path, '_metrics.csv'), index=True)

coeffs_df = pd.DataFrame(coeff_list)
coeffs_df.to_csv(os.path.join(local_save_path, '_coeffs.csv'), index=True)