"""
calculate_effects.py
Author: Maggie Jacoby
Date: 2021-04-26

"""


import os
import sys
import csv
import argparse
import itertools
import numpy as np
import pandas as pd
from glob import glob


def subset_df(df):
    hubs = [col for col in df.columns if f'RS' in col or f'BS' in col]
    hub_df = df[hubs]
    metrics = ['Accuracy', 'F1', 'F1 neg', 'tnr', 'fpr', 'fnr', 'tpr', 'tn', 'fp', 'fn', 'tp']
    metric_df = df[metrics]
    return hub_df, metric_df



def get_effects(hub_df, metric_df):
    full_metrics = []
    div = len(metric_df)/2
    for metric in metric_df.columns:
        new_df = hub_df.multiply(metric_df[metric], axis='index')
        sum_col = new_df.sum(axis=0)/div
        sum_col.rename(metric, inplace=True)
        full_metrics.append(sum_col)
    full_metrics = pd.concat(full_metrics, axis=1)
    return full_metrics



def get_interactions(hub_df, level=2):
    hubs = hub_df.columns
    interactions = list(itertools.combinations(hubs, level))
    interaction_df = []
    for hub_set in interactions:
        col_name = ''
        H1, H2 = hub_set[0], hub_set[1]
        idx = range(1,len(hub_df)+1)
        new_col = pd.Series([1.0]*len(hub_df), index=idx)
        

        for hub in hub_set:
            col_name += hub + 'x'
            mult_col = hub_df[hub]
            new_col.index=mult_col.index
            new_col = new_col.multiply(mult_col)

        col_name = col_name.strip('x')
        new_col.rename(col_name, inplace=True)
        interaction_df.append(new_col)

    interaction_df = pd.concat(interaction_df, axis=1)
    return interaction_df




def avg_effect(DF):
    BL = DF.head(1)
    BL.index=['BL']
    df = DF.tail(len(DF)-1)
    avg_dict = {}
    div = len(df)
    for metric in df.columns:
        avg_dict[metric] = df[metric].sum()/div
    avg_df = pd.DataFrame(avg_dict, index=['avg'])
    df_ = pd.concat([BL, avg_df])
    return df_




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate effects from factorial analysis output')

    parser.add_argument('-path','--path', default='', type=str, help='path of stored data')
    args = parser.parse_args()

    file_path = args.path
    run_comparison = os.path.basename(file_path.strip('/') )
    print(run_comparison)

    run_paths = glob(os.path.join(file_path, 'noCV_*'))
    for run in run_paths:
        print(run)

        results = glob(os.path.join(run, '*_ffa_*.csv'))

        for h_path in sorted(results):
            home = os.path.basename(h_path).split('_')[0]
            ffa_output = pd.read_csv(h_path, index_col='run')

            hub_df, metric_df = subset_df(ffa_output)

            main_ef = get_effects(hub_df, metric_df)
            twoFI = get_interactions(hub_df)
            two_ef = get_effects(twoFI, metric_df)

            avg = avg_effect(metric_df)
            # print(avg)
            full_effects = pd.concat([avg, main_ef, two_ef], axis=0)
            # break

            save_folder = os.path.join(run, 'fa_results')
            os.makedirs(save_folder, exist_ok=True)
            full_effects.to_csv(os.path.join(save_folder, f'{home}_{os.path.basename(run.strip("/") )}.csv'))


    
