"""
train_save_model.py
author: Maggie Jacoby
Date: 2021-07-28
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, date
import random

from etl import ETL
from train import TrainModel


param_dict = dict(
    C = 0.3, 
    penalty = 'l1', 
    min_inc = 5,
    lag_type = 'avg',
    grp_len = 8,
    lag = 8
)

grps_by_home = {'H1': [5,4], 'H2':[1,0], 'H3':[3,0], 'H5':[0,1], 'H6':[1,4]}


def get_balanced_df(all_homes, groups):
    """ read in groups from each home and create a balanced df for training
    """
    print('creating df...')
    train_data = []
    for x in all_homes:
        grp_choices = groups[x]
        grps = [all_homes[x].daysets[k] for k in grp_choices]
        train_data.extend(grps)
    train_df = pd.concat(train_data)
    print(len(train_df))
    return train_df


def train_new(train_df, params, store_dir, H='all'):
    """ take in a param dict and training df and train a new model"""
    print('training model ... ')

    train_all = TrainModel(
        H_num=H,
        train_data=train_df, 
        C=params['C'],
        penalty=params['penalty'],
        lag=params['lag']
        )
    coeffs = train_all.coeffs
    coeffs.to_csv(os.path.join(store_dir, 'coefs_saved_model.csv'))
    print(coeffs)

    return train_all.model


def save_model(model, store_dir, model_name=''):
    """Saves model as a pickle object and return nothing
    """
    print('saving model.... ')

    save_name = os.path.join(store_dir, f'{model_name}.pickle')

    if not os.path.isfile(save_name):
        pickle.dump(model, open(save_name, 'wb'))
        print(f'\t>>> Writing model to {save_name}')
    else:
        pickle.dump(model, open(save_name, 'wb'))
        print(f'\t>>> Model {save_name} exists. Overwriting previous')



def generate_datasets(homes, params):
    print('generating data... ')
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


store_dir = f'/Users/maggie/Desktop/saved_final'
os.makedirs(store_dir, exist_ok=True)

home_data = generate_datasets(homes=homes, params=param_dict)
train_df = get_balanced_df(all_homes=home_data, groups=grps_by_home)
model = train_new(train_df=train_df, params=param_dict, H='all', store_dir=store_dir)

save_model(model=model, model_name=f'ARLR_test', store_dir=store_dir)




