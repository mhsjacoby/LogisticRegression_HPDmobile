"""
add_groundtruth.py
"""


import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, date

parent_dir = os.path.dirname(os.getcwd())

def replace_occ(H_num):

    occ_data = glob(os.path.join(parent_dir, 'raw_data_files', H_num, f'{H_num}-*_occupancy.csv'))[0]
    occ_df = pd.read_csv(occ_data, index_col='timestamp')
    occ_df['occ max'] = occ_df[occ_df.columns.difference(['occupied'])].max(axis=1)
    occ_df.index = pd.to_datetime(occ_df.index)

    datafiles = glob(os.path.join(parent_dir, 'raw_data_files', H_num, f'{H_num}*_prob.csv'))
    hub_diffs = []

    for hub_file in sorted(datafiles):

        fname = os.path.basename(hub_file).replace('_prob', '')
        hub_df = pd.read_csv(hub_file, index_col='timestamp')
        hub_df.index = pd.to_datetime(hub_df.index)
        
        merged_df = hub_df.merge(occ_df['occ max'], how='inner', on='timestamp')

        diff_df = merged_df[['occupied', 'occ max']].diff(axis=1)[['occ max']]
        diff_df.columns = [fname.rstrip('.csv').lstrip(H_num)]
        hub_diffs.append(diff_df)

        merged_df['occupied'] = merged_df['occ max']
        merged_df.drop(columns=['occ max'], inplace=True)

        merged_df.to_csv(os.path.join(parent_dir, 'raw_data_files', H_num, fname))

    diff = pd.concat(hub_diffs, axis=1)
    diff.to_csv(os.path.join(parent_dir, 'Results', f'{H_num}_occ_differences.csv'))


homes = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
for home in homes:
    replace_occ(H_num=home)
