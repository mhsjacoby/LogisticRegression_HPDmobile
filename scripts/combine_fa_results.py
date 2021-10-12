"""
combine_fa_results.py
Author: Maggie Jacoby
Date: 2021-07-28


To-Do:
- normalize results
"""


import os
import sys
import yaml
import csv
import argparse
import itertools
import numpy as np
import pandas as pd
from glob import glob



def read_config(home):

    config_dir = '/Users/maggie/Documents/Github/LogisticRegression_HPDmobile/configuration_files'
    config_file_path = os.path.join(config_dir, f'{home}_etl_config.yml')
    with open(config_file_path) as f:
        config = yaml.safe_load(f)

    return config



def subset_df(df, hubs):

    var_df = df.loc['run var']
    var_df = var_df .dropna(axis=0)
    var_df = pd.DataFrame(data=var_df).transpose()
    sd_df = var_df.apply(np.sqrt)
    sd_df = sd_df.rename(index={'run var': 'std dev'})

    df.drop(['run var'], inplace=True)

    hub_df = df[hubs]
    metrics = [col for col in df.columns if col not in hubs]
    metric_df = df[metrics]
    return hub_df, metric_df, sd_df



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




def avg_effect(df):
    avg_dict = {}
    div = len(df)
    for metric in df.columns:
        # avg = df[metric].sum(axis=1)
        avg_dict[metric] = df[metric].sum()/div
    avg_df = pd.DataFrame(avg_dict, index=['avg'])
    return avg_df


store_dir = '/Users/maggie/Desktop/FA_withARLR_results/FA_metrics'

run_folder = '/Users/maggie/Desktop/completed_runs'

if __name__ == '__main__':

    all_files = glob(os.path.join(run_folder, 'H3*-env.csv'))
    print(all_files)


    for file_path in all_files:
        # file_path = 
        print(f'reading file: {file_path}')
        # sys.exit()

        home, comparison = os.path.basename(file_path).split('_')

        ffa_output = pd.read_csv(file_path, index_col='Run')
        config = read_config(home)
        color = config['H_system'][0]
        hubs = [f'{color.upper()}S{n}' for n in config['hubs']]

        hub_df, metric_df, sd_df = subset_df(ffa_output, hubs)
        Main_effects = get_effects(hub_df, metric_df)
        # Main_effects.to_csv(os.path.join(store_dir, f'{home}_main_{comparison}.csv'))

        TwoFI = get_interactions(hub_df)
        Two_level_effects = get_effects(TwoFI, metric_df)
        Two_level_effects.to_csv(os.path.join(store_dir, '2FI', f'{home}_2FI_{comparison}'))

        ThreeFI = get_interactions(hub_df, level=3)
        Three_level_effects = get_effects(ThreeFI, metric_df)
        Three_level_effects.to_csv(os.path.join(store_dir, '3FI', f'{home}_3I_{comparison}'))
        # full_effects = pd.concat([Main_effects, Two_level_effects, Three_level_effects], axis=0)

        FourFI = get_interactions(hub_df, level=4)
        Four_level_effects = get_effects(FourFI, metric_df)
        Four_level_effects.to_csv(os.path.join(store_dir, '4FI', f'{home}_4I_{comparison}'))

        FiveFI = get_interactions(hub_df, level=5)
        Five_level_effects = get_effects(FiveFI, metric_df)
        Five_level_effects.to_csv(os.path.join(store_dir, '5FI', f'{home}_5I_{comparison}'))

        # avg = avg_effect(metric_df)
        # full_wavg = pd.concat([avg, Main_effects, Two_level_effects, sd_df])

        # full_wavg = full_wavg[['Accuracy', 'F1', 'F1 neg', 'tnr', 'fnr', 'tpr', 'fpr', 
                                # 'tn', 'fp', 'fn', 'tp', 'RMSE', 'MCC']]
        # full_wavg.to_csv(os.path.join(store_dir, f'{home}_{comparison}_3fi'))
        # sys.exit()

        