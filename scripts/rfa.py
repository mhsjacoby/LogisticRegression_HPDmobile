"""
rfa.py (run_factorial_analysis)
"""

import os
import sys
import csv
# import json
import argparse
# import itertools
import numpy as np
import pandas as pd
from glob import glob
from functools import reduce

# from datetime import datetime, timedelta, time

from etl import ETL
from trainAR import TrainModel
from testAR import TestModel




class FFA_instance(ETL):
    
    def __init__(self, run, hubs, home, mods, days, fill='zeros', env=True):

        self.run  = run[0]
        self.spec = run[1]
        self.hubs = hubs
        self.H_num = home
        self.mod_dict = mods
        self.days = days
        self.fill_type=fill
        self.env = env

        ## Method in ETL parent class
        self.get_directories()  

        self.run_modalities = self.get_hub_modalities()
        self.df = self.create_df()        

    def get_hub_modalities(self):
        run_mods = {}
        for x,y in zip(self.hubs, self.spec):
            run_mods[x] = self.mod_dict[y]
        return run_mods


    def get_null_prediction(self):
        null_dfs = []
        for mod in self.mod_dict.values():
            # if mod is None:
            if mod == 'None':
                continue
            null_df = self.get_data(hub=self.hubs[0], mod=mod)
            null_df[mod] = 0
            null_dfs.append(null_df)
        return null_dfs

    
    def get_usecols(self, mod):
        if self.env:
            cols = ['timestamp', mod, 'temp', 'rh', 'light', 'co2eq', 'occupied']
        else:
            cols = ['timestamp', mod, 'occupied']
        return cols


        

    def get_data(self, hub, mod, resample_rate='5min', thresh=0.5):
        data_path = os.path.join(self.raw_data, self.H_num, f'{self.H_num}{hub}.csv')
        cols = self.get_usecols(mod)
        df = pd.read_csv(data_path, index_col='timestamp', usecols=cols)
        df.index = pd.to_datetime(df.index)
        df = df.resample(rule=resample_rate).mean()
        df['occupied'] = df['occupied'].apply(lambda x: 1 if (x >= thresh) else 0)
        df = self.fill_df(df=df)

        return df

    
    def create_df(self):
        all_hub_dfs = []
        
        for hub, mod in self.run_modalities.items():

            if mod == 'None':
                continue

            hub_df = self.get_data(hub=hub, mod=mod)
            all_hub_dfs.append(hub_df)
        
        null_predictions = self.get_null_prediction()
        all_hub_dfs.extend(null_predictions)

        df = pd.concat(all_hub_dfs).groupby(level=0).max()

        df = self.create_HOD(df)
        df = self.create_rolling_lags(df)

        self.train, self.test = self.get_train_test(df)

        return



def fracfact(n, level='full'):
    spec_path = os.path.join(os.getcwd(), 'raw_data_files', 'fracfact_output')
    run_file = os.path.join(spec_path, f'{n}_hub_{level}.csv')
    
    run_specifications = []
    with open(run_file) as ff_out:
        for i, row in enumerate(csv.reader(ff_out), 1):
            run_specifications.append((i, row))

    return run_specifications


def create_mod_dict(comparison):
    c_pos, c_neg = comparison.split('_')
    mod_dict = {'-1': c_neg, '1': c_pos}
    return mod_dict

# img_audio = {1:'Image', -1:'Audio'},
# img_None = {1:'Image', -1:'NA'},
# audio_None = {1:'Audio', -1:'NA'}



def set_run(home, data, run, spec, metrics, coeffs, cv):

    Train = TrainModel(H_num=home, train_data=data.train, cv=cv)
    Test = TestModel(H_num=home, test_data=data.test, model=Train.model)
    Test.test_AR(clf=Train.model)
    
    Test.pred_metrics['run'] = run
    Test.pred_metrics['inclusion'] = spec
    metrics[run] = Test.pred_metrics

    Train.coeffs['run'] = run
    Train.coeffs['inclusion'] = spec
    coeffs[run] = Train.coeffs

    return metrics, coeffs


def merge_dfs(df_list):
    df2 = pd.DataFrame(df_list).transpose()
    df = pd.DataFrame(df2['inclusion'].to_list(), columns=hubs)
    df.index = df2.index
    df = df.merge(df2, left_index=True, right_index=True)
    df.index = df['run']
    df.drop(columns=['inclusion', 'run'], inplace=True)

    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Running factorial analysis')

    parser.add_argument('-home', '--home', default='H1', type=str, help='Home to get data for, eg H1')
    parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')
    parser.add_argument('-level', '--level', type=str, default='full')
    # parser.add_argument('-cv', '--cv', default=False, action='store_true', help='Perform cross-validation?')
    # parser.add_argument('-fname', '--fname', default='', type=str, help='name to append to saved files')
    # parser.add_argument('-compare', '--compare', type=str, default='img_audio')
    # parser.add_argument('-env', '--env', default=False, action='store_true', help='Use env in predictions?')
    args = parser.parse_args()

    fill_type = args.fill_type
    run_level = args.level
    # cv = False if not args.cv else True
    # env = False if not args.env else True
    # fname = args.fname
    # compare = args.compare



    # cv_ = [True, False]
    env_ = [True, False]
    compare_ = ['img_audio', 'audio_None', 'img_None']

    cv_ = [False]
    for compare in compare_[1:]:
        for cv in cv_:
            for env in env_:
                print(f'===== env: {env}, cv: {cv}, compare: {compare} =====')

                homes = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
                fname_cv = '' if cv else 'no'
                fname_env = '' if env else 'no'
                fname_compare = ''.join([x[0] for x in compare.split('_')])

                fname = f'{fname_cv}CV_{fname_env}ENV_{run_level}_{fname_compare}'
                save_dir = os.path.join('/Users/maggie/Desktop/FFA_results', compare, fname)
                os.makedirs(save_dir, exist_ok=True)

                for home in homes:
                    H = ETL(H_num=home, fill_type=fill_type)

                    H.read_config()
                    hubs = H.get_hubs()
                    days = H.get_days()
                    n_hubs = len(hubs)
                    run_specs = fracfact(n=n_hubs, level=run_level)
                    md = create_mod_dict(compare)

                    metrics = {}
                    coeffs = {}

                    # Get baseline
                    baseline = ETL(H_num=home, fill_type=fill_type) 
                    baseline.generate_dataset()
                    if not env:
                        print('not including env in predictions')
                        baseline.train.drop(columns=['co2eq', 'light', 'rh', 'temp'], inplace=True)
                        baseline.test.drop(columns=['co2eq', 'light', 'rh', 'temp'], inplace=True)
                    bl_spec = [0 for i in range(0,len(hubs))]

                    metrics, coeffs = set_run(run=0, home=home, data=baseline, spec=bl_spec, metrics=metrics, coeffs=coeffs, cv=cv)

                    for i in run_specs:

                        inst = FFA_instance(run=i, hubs=hubs, home=home, mods=md, fill=fill_type, days=days, env=env)
                        metrics, coeffs = set_run(home=home, data=inst, run=inst.run, spec=inst.spec, metrics=metrics, coeffs=coeffs, cv=cv)


                    metrics_df = merge_dfs(metrics)
                    metrics_df.to_csv(os.path.join(save_dir, f'{home}_ffa_{fname}.csv'))

                    coef_df = merge_dfs(coeffs)
                    coef_df.to_csv(os.path.join(save_dir, f'{home}_coefs_{fname}.csv'))