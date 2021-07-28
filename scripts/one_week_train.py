"""
one_week_train.py
author: Maggie Jacoby
date: 2021-07-21
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


param_dict = dict(
    C = 0.1, 
    penalty = 'l1', 
    min_inc = 5,
    lag_type = 'avg',
    grp_len = 8,
    lag = 8
)

def set_params(change_item, value, param_dict=param_dict):
    param_dict[change_item] = value
    return param_dict

def attach_params(df, params):
    for k,v in params.items():
        df[k] = v
    return df

def test_combined(all_homes, params):
    metrics = []
    coeffs = {}

    for h in sorted(all_homes, reverse=True):
        
        train_single = all_homes[h]

        for i in range(0,len(train_single.daysets)):
            self_groups = list(train_single.daysets)
            train_self = self_groups.pop(i)

            single = TrainModel(
                H_num=h, 
                train_data=train_self, 
                C=params['C'],
                penalty=params['penalty'],
                )

            coeffs[f'train {h} {i}'] = single.coeffs

            if len(self_groups) < 1:
                print(f'!!!!!!!!!!!!!!!! NOT ENOUGH GROUPS IN {h} !!!!!!!!!!!!!')
                continue
            for j in range(0,len(self_groups)):
                test_self = self_groups[j]
                self_test = TestModel(
                    H_num=h, 
                    model=single.model, 
                    non_prob=single.non_prob, 
                    test_data=test_self,
                    )

                self_test.metrics[f'train home']= h
                self_test.metrics[f'test home']= h
                self_test.metrics['self/cross'] = 'self'
                metrics.append(self_test.metrics)

            all_test_homes = [x for x in all_homes if x != h]

            for test_home in all_test_homes:
                cross_home = all_homes[test_home]
                cross_groups = cross_home.daysets
                
                for test_grp in cross_groups:
                    cross_test = TestModel(
                        H_num=h, 
                        model=single.model, 
                        non_prob=single.non_prob, 
                        test_data=test_grp,
                        )

                    cross_test.metrics[f'train home']= h
                    cross_test.metrics[f'test home']= test_home
                    cross_test.metrics['self/cross'] = 'cross'
                    metrics.append(cross_test.metrics)
            
    metrics_df = pd.concat(metrics)
    coeffs_df = pd.DataFrame(coeffs)
    return coeffs_df, metrics_df



def get_classifier_summaries(df):

    df['Classifier'] = df.index
    df['perc occ'] = (df['tp'] + df['fn']) / (df['tp'] + df['fn'] + df['fp'] + df['tn']) 
    cat_cols = ['Classifier', 'self/cross', 'test home']
    num_cols = ['Accuracy', 'F1', 'F1 neg', 'perc occ']
    metrics = pd.DataFrame()
    metrics[num_cols] = df[num_cols].apply(pd.to_numeric)
    metrics[cat_cols] = df[cat_cols]

    byhome_mean = metrics.groupby(cat_cols).mean()
    byhome_std = metrics.groupby(cat_cols).std()
    byhome_std.columns = [f'{x} std' for x in byhome_std.columns]
    by_home = pd.concat([byhome_mean, byhome_std], axis=1)
    by_home = by_home.reset_index(level=['self/cross', 'test home'] )

    grpby_mean = metrics.groupby(['Classifier', 'self/cross']).mean()
    grpby_std = metrics.groupby(['Classifier', 'self/cross']).std()
    grpby_std.columns = [f'{x} std' for x in grpby_std.columns]
    full = pd.concat([grpby_mean, grpby_std], axis=1)
    full = full.reset_index(level=['self/cross'] )

    return full, by_home



def generate_datasets(homes, params):
    all_homes = {}
    for h in homes:
        x = ETL(
            H_num=h,
            lag=params['lag'],
            min_inc=params['min_inc'],
            lag_type=params['lag_type'],
            grp_len=params['grp_len']
            )

        x.generate_dataset()
        all_homes[h] = x
    return all_homes


homes = ['H1', 'H2', 'H3', 'H5', 'H6']
# homes = ['H5', 'H2'] 

change_item = 'train_len'

store_dir = f'/Users/maggie/Desktop/all_results/change_{change_item}'
print(store_dir)
os.makedirs(store_dir, exist_ok=True)
    
parameters_set = param_dict
fname = f'{change_item}_1'

home_data = generate_datasets(homes, params=parameters_set)
for home in home_data:
    print('----------------------------')
    print(home)
    daysets = home_data[home].daysets
    print('>>>', len(daysets))
    for ds in daysets:
        print('>', len(ds.day.unique()))

sys.exit()
full_coeffs, full_metrics = test_combined(all_homes=home_data, params=parameters_set)
full_coeffs.to_csv(os.path.join(store_dir, f'coeffs_{fname}.csv'))
full_metrics.to_csv(os.path.join(store_dir, f'metrics_{fname}.csv'))

full, by_home = get_classifier_summaries(full_metrics)

by_home.to_csv(os.path.join(store_dir, f'home_gpby_{fname}.csv'))
full.to_csv(os.path.join(store_dir, f'full_gpby_{fname}.csv'))
full['test home'] = 'H0'

cols = ['self/cross', 'test home', 'Accuracy', 'F1', 'F1 neg', 'Accuracy std']

group_results = full.loc[['AR predictions','NP'], cols]
home_results = by_home.loc[['AR predictions','NP'], cols]

joined_results = pd.concat([group_results, home_results])
joined_results = attach_params(df=joined_results, params=parameters_set)
joined_results.to_csv(f'~/Desktop/all_results/_final_{change_item}.csv')