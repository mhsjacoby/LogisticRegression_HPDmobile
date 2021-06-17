"""
run_ffa.py
Author: Maggie Jacoby
Edited: 2020-10-20 - calculate variance of runs

These classes are used to generate objects for each home, and individual FFA runs. 
Import into Load_data_into_pstgres notebook or run stand alone (preferred).
"""

import os
import sys
import csv
import json
import argparse
import itertools
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timedelta, time

from functools import reduce
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from my_functions import *
from pg_functions import *

start_end_file = 'start_end_dates.json'

"""audio, env, img, None
"""

class Home():
    
    def __init__(self, pg, system, level, threshold='0.8', schema='public'):
        self.schema = schema
        self.pg = pg
        self.system = system.lower().split('-')
        self.pg_system = pg.home
        self.level = level
        self.hubs = self.get_distinct_from_DB('hub')
        self.days = get_date_list(read_file=start_end_file, H_num=H_num)
        # self.start_time, self.end_time = self.get_hours(system)
        self.run_specifications = self.get_FFA_output()

    # def get_days(self, threshold, system):
    #     all_days_db = [x.strftime('%Y-%m-%d') for x in self.get_distinct_from_DB('day')]
    #     days_above_file = os.path.join('/Users/maggie/Desktop/CompleteSummaries', f'all_days_above_{threshold}.json')
    #     with open(days_above_file) as file:
    #         fdata = json.load(file)
        
    #     all_days_th = fdata[system]
    #     days = sorted(list(set(all_days_db).intersection(all_days_th)))
    #     print(f'Number of days to available above threshold {threshold}: {len(days)}')
    #     return days

    # def get_hours(self, system):
    #     hours_asleep_file = '/Users/maggie/Desktop/CompleteSummaries/hours_asleep.json'
    #     with open(hours_asleep_file) as file:
    #         fdata = json.load(file)
    #     times = fdata[system]
    #     start_time = datetime.strptime(times['start'], '%H:%M:%S').time()
    #     end_time = datetime.strptime(times['end'], '%H:%M:%S').time()
    #     print(f'Not calculating between the hours of {start_time} and {end_time}')
    #     return start_time, end_time


    def get_distinct_from_DB(self, col):
        print(self.pg_system)

        query = """
            SELECT DISTINCT %s
            FROM %s.%s_inf;
            """ %(col, self.schema, self.pg_system)

        distinct = self.pg.query_db(query)[col].unique()
        return sorted(distinct)
    
    
    def select_from_hub(self, hub, mod):

        select_query = """
            SELECT day, hr_min_sec, hub, %s, occupied
            FROM %s.%s_inf
            WHERE %s_inf.hub = '%s'
            """ %(mod, self.schema, self.pg_system, self.pg_system, hub)

        return select_query


    def get_FFA_output(self):
        spec_path = f'/Users/maggie/Documents/Github/HPD-Inference_and_Processing/SensorFusion/fracfact_output'
        num_hubs = len(self.hubs)
        run_file = os.path.join(spec_path, f'{num_hubs}_hub_{self.level}.csv')

        run_specifications = []
        with open(run_file) as FFA_output:
            for i, row in enumerate(csv.reader(FFA_output), 1):
                run_specifications.append((i, row))
        return run_specifications


# def get_counts(df):
#     skip_cols = ['day', 'hr_min_sec', 'hub']

#     for col in df[df.columns.difference(skip_cols)]:
#         print(df[col].value_counts())

# def cos_win(min_win=.25, max_win=3, df_len=8640):
#     min_win = min_win * 360
#     max_win = max_win * 360

#     win_range = max_win - min_win
#     t = np.linspace(0, df_len, df_len)
#     win_lim = np.round(win_range/2 * np.cos(t*2*np.pi/8640) + win_range/2 + min_win).astype(int)
#     return win_lim

# def get_forward_pred(data):
#     df = data.copy()
    
#     time_window = cos_win(min_win=.25, max_win=2, df_len=len(df))
#     ind_map = {x:y for y,x in zip(time_window, df.index)}   # x is the index number of the df, y is the lookahead value
    
#     skip_cols = ['day', 'hr_min_sec', 'hub', 'occupied']
    
#     for col in df[df.columns.difference(skip_cols)]:
#         ind_list = df.loc[df[col] == 1].index

#         for idx in ind_list:
#             j = idx + ind_map[idx]
#             df.loc[(df.index >= idx) & (df.index <= j), col] = 1
    
#     return df
        # print(df[col].value_counts())
        # print(f'setting {len(ind_list)} indices in column: {col}')

class FFA_instance():
    # mod_dict = {'-1': 'audio', '1': 'img'}
    
    def __init__(self, run, Home, comparison):
        self.Home = Home
        # self.mod_dict = self.define_comparison(comparison)
        self.mod_dict = self.create_mod_dict(comparison)
        self.run  = run[0]
        self.spec = run[1]
        # self.check_spec()
        self.run_modalities = self.get_hub_modalities()
        self.df = self.create_df()
        self.predictions = self.get_predictions()
        self.results_by_day = {}
        self.rate_results = self.test_days(days=self.Home.days)
        
        self.TPR, self.FPR = np.mean(self.rate_results['TPR']), np.mean(self.rate_results['FPR'])
        self.TNR, self.FNR = np.mean(self.rate_results['TNR']), np.mean(self.rate_results['FNR']) 
        self.f1, self.accuracy = np.mean(self.rate_results['f1']), np.mean(self.rate_results['accuracy'])

        self.var_TPR, self.var_FPR = np.var(self.rate_results['TPR']), np.var(self.rate_results['FPR'])
        self.var_TNR, self.var_FNR = np.var(self.rate_results['TNR']), np.var(self.rate_results['FNR']) 
        self.var_f1, self.var_accuracy = np.var(self.rate_results['f1']), np.var(self.rate_results['accuracy'])

    def create_mod_dict(self, comparison):
        comp_pos1, comp_neg1 = comparison.split('_')
        mod_dict = {'-1': comp_neg1, '1': comp_pos1}
        print(mod_dict)
        return mod_dict
        

    def get_hub_modalities(self):
        run_mods = {}
        for x,y in zip(self.Home.hubs, self.spec):
            run_mods[x] = self.mod_dict[y]
        return run_mods


    def get_null_prediction(self):
        hub = self.Home.hubs[0]
        mod = self.mod_dict['1']
        print(hub, mod)
        null_df = self.Home.pg.query_db(self.Home.select_from_hub(hub, mod))
        null_df.drop(columns=['hub', mod], inplace=True)
        null_df['null'] = 1
        return null_df
    
    def create_df(self):
        df_list = []
        print(self.run_modalities)
        for hub in self.run_modalities:
            changes = {}
            mod = self.run_modalities[hub]
            print(f'hub: {hub}, modality: {mod}')
            if mod == 'None':
                continue
            hub_df = self.Home.pg.query_db(self.Home.select_from_hub(hub, mod))

            hub_df.drop(columns=['hub'], inplace=True)
            rename_cols = {mod:f'{mod}_{hub[2]}'}
            hub_df.rename(columns = rename_cols, inplace=True)
            df_list.append(hub_df)
        
        if len(df_list) == 0:
            df_merged = self.get_null_prediction()
            print('null!')

        else:
            df_merged = reduce(lambda left, right: pd.merge(left, right, on=['day', 'hr_min_sec', 'occupied'], how='outer'), df_list)
            print('not null')

        print(len(df_merged))
        col = df_merged.pop('occupied')
        df_merged.insert(len(df_merged.columns), col.name, col)
        
        changes = {}
        print(df_merged.columns)
        skip_cols = ['day', 'hr_min_sec', 'occupied']
    
        for hub_mod in df_merged.columns.difference(skip_cols):
            changes[f'{hub_mod}-before'] = df_merged[hub_mod].value_counts()

        df_merged = df_merged[(df_merged != -1).all(axis=1)]

        for hub_mod in df_merged.columns.difference(skip_cols):
            changes[f'{hub_mod}-after'] = df_merged[hub_mod].value_counts()

        changes_df = pd.DataFrame.from_dict(changes).fillna(0).astype(int)
        changes_df = changes_df.transpose().sort_index()
        changes_df['total'] = changes_df[list(changes_df.columns)].sum(axis=1)
        print(changes_df)
        return df_merged

    
    def get_predictions(self):
        df = self.df.copy()
        skip_cols = ['day', 'hr_min_sec', 'occupied']
        df['prediction'] = 0

        df.loc[df[df.columns.difference(skip_cols)].sum(axis=1) > 0, 'prediction'] = 1
        
        for col in df[df.columns.difference(skip_cols)]:
            print(df[col].value_counts())
        print(df['occupied'].value_counts())

        skip_cols.append('prediction')
        return df[skip_cols]
        

    def test_days(self, days):
        TPR, FPR, TNR, FNR, f1, acc = [], [], [], [], [], []
        
        for day_str in sorted(days):
            day = datetime.strptime(day_str, '%Y-%m-%d').date()
            day_df = self.predictions.loc[self.predictions['day'] == day]
            # day_df = day_df.loc[~((day_df['hr_min_sec'] >= self.Home.start_time) & (day_df['hr_min_sec'] < self.Home.end_time))]

            tn, fp, fn, tp = confusion_matrix(day_df['occupied'], day_df['prediction'], labels=[0,1]).ravel()
            f1.append(f1_score(day_df['occupied'], day_df['prediction']))
            acc.append(accuracy_score(day_df['occupied'], day_df['prediction']))
            self.results_by_day[day_str] = (tn, fp, fn, tp)

            tpr = tp/(tp+fn) if tp+fn > 0 else 0.0
            fpr = fp/(tn+fp) if tn+fp > 0 else 0.0

            tnr = tn/(tn+fp) if tn+fp > 0 else 0.0
            fnr = fn/(tp+fn) if tp+fn > 0 else 0.0

            TPR.append(float(f'{tpr:.4}'))
            FPR.append(float(f'{fpr:.4}'))

            TNR.append(float(f'{tnr:.4}'))
            FNR.append(float(f'{fnr:.4}'))

        return {'TPR': TPR, 'FPR': FPR, 'TNR': TNR, 'FNR': FNR, 'f1': f1, 'accuracy': acc}


def get_instances(H, comparison):

    d = {
        'Run': [], 'Inclusion': [], 'Name': [],
        'False Positive Rate': [], 'True Positive Rate': [],
        'False Negative Rate': [], 'True Negative Rate': [],
        'F1-Score': [], 'Accuracy': []
        }

    V = {
        'False Positive Rate': [], 'True Positive Rate': [],
        'False Negative Rate': [], 'True Negative Rate': [],
        'F1-Score': [], 'Accuracy': []
        }


    all_instances = {}

    for x in H.run_specifications:
        
        inst = FFA_instance(x, H, comparison)

        all_instances[inst.run] = inst
        
        d['False Positive Rate'].append(inst.FPR)
        d['True Positive Rate'].append(inst.TPR)
        d['False Negative Rate'].append(inst.FNR)
        d['True Negative Rate'].append(inst.TNR)
        
        d['Run'].append(inst.run)
        d['Inclusion'].append(inst.spec)
        d['F1-Score'].append(inst.f1)
        d['Accuracy'].append(inst.accuracy)
        d['Name'].append(f'Run {inst.run}: {inst.run_modalities}')


        V['False Positive Rate'].append(inst.var_FPR)
        V['True Positive Rate'].append(inst.var_TPR)
        V['False Negative Rate'].append(inst.var_FNR)
        V['True Negative Rate'].append(inst.var_TNR)
        V['F1-Score'].append(inst.var_f1)
        V['Accuracy'].append(inst.var_accuracy) 

    N_runs = len(H.run_specifications)
    N_days = len(H.days)

    SE = {}
    for v in V:
        SE[v] = np.sqrt(np.mean(V[v])*(N_days/N_runs))/10

    SE_df = pd.DataFrame(SE, index=['SE'])
    roc_df = pd.DataFrame(d)

    return roc_df, SE_df




"start_end_dates.json"

if __name__=='__main__':

    # """ For running in bash script """
    run_level = "full"
    schema = "nofill"

    home_system = sys.argv[1]
    H_num, color = home_system.split('-')
    comparison = sys.argv[2]
    comp = comparison.split('_')

    """ For running in terminal """
    # parser = argparse.ArgumentParser(description="Description")

    # parser.add_argument('-system', '--system', type=str)
    # parser.add_argument('-level', '--level', type=str, default='full')
    # parser.add_argument('-compare', '--compare', type=str, default='image_audio')
    # args = parser.parse_args()

    # home_system = args.system
    # H_num, color = home_system.split('-')
    # run_level = args.level
    # comparison = args.compare
    # comp = comparison.split('_')



    print(f"***** Running: {home_system}, {comparison}")

    home_parameters = {'home': f'{H_num.lower()}_{color}'}
    pg = PostgreSQL(home_parameters, schema=schema)

    H = Home(pg=pg, system=home_system, level=run_level, schema=schema)

    roc_df, SE = get_instances(H, comparison)
    df2 = pd.DataFrame(roc_df['Inclusion'].to_list(), columns=H.hubs)

    df2.index = roc_df.index
    df2 = df2.merge(roc_df, left_index=True, right_index=True)
    df2.index = df2['Run']
    df2.drop(columns=['Inclusion', 'Run'], inplace=True)
    dfwSE = df2.append(SE, ignore_index=False)
    dfwSE.index.rename('Run')

    
    dfwSE.to_csv(f'/Users/maggie/Desktop/FFA_output/{home_system}_{run_level}_{comp[0]}-{comp[1]}.csv', index_label='Run')