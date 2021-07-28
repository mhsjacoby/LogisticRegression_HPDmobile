"""
get_results.py
author: Maggie Jacoby
date: 2021-07-16
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
from test import TestModel
import statsmodels.api as sm

param_dict = dict(
    C = 0.3, 
    penalty = 'l1', 
    min_inc = 5,
    lag_type = 'avg',
    grp_len = 8,
    lag = 8
)

def set_params(change_item, value, param_dict=param_dict):
    param_dict[change_item] = value
    return param_dict

def get_fname(param_list):
    fname = [x[1] for x in param_list.items()]
    fname = [str(x).replace('.','') for x in fname]
    fname = '_'.join(fname)
    print(f'Saving to file: {fname}')
    return fname

def attach_params(df, params):
    for k,v in params.items():
        df[k] = v
    return df

def train_statsmodel(df, fname):
    y = df['occupied']
    X = df[df.columns.difference(['occupied'], sort=False)]
    X = X.drop(columns = ['day'])
    X['intercept'] = 1
    sm_model = sm.Logit(y, X).fit_regularized(method='l1', alpha=3.3)
    summary = sm_model.summary()
    print(summary)

    f1 = open(f'/Users/maggie/Desktop/coeffs_{fname}_s2.csv','w')
    f1.write(summary.as_csv())
    f1.close()

    f2 = open(f'/Users/maggie/Desktop/coeffs_{fname}_s2.tex','w')
    f2.write(summary.as_latex())
    f2.close()
    sys.exit()
    return summary



def get_balanced_df(all_homes, params, fname=''):
    metrics = []
    coeffs = {}
    grp_choice_names = {}
    grps_by_home = {'H1': [5,4], 'H2':[1,0], 'H3':[3,0], 'H5':[0,1], 'H6':[1,4]}

    # for i in range(0,10):
    train_data = []
    for x in all_homes:
        # grp_choices = random.sample(range(0,len(all_homes[x].daysets)), k=2)
        # print(f'picking {grp_choices} for house {x}')
        # grp_choice_names[x] = grp_choices
        grp_choices = grps_by_home[x]
        grps = [all_homes[x].daysets[k] for k in grp_choices]
        train_data.extend(grps)
    train_df = pd.concat(train_data)
    
    train_all = TrainModel(
        H_num=f'all',
        train_data=train_df, 
        C=params['C'],
        penalty=params['penalty'],
        lag=params['lag']
        )
    coeffs = train_all.coeffs
    coeffs.to_csv(f'/Users/maggie/Desktop/coefs_saved_model.csv')
    # pd.DataFrame(grp_choice_names).to_csv(f'/Users/maggie/Desktop/groups_{fname}.csv')
    s = train_statsmodel(df=train_df, fname=fname)

    for h in all_homes:
        test_data = all_homes[h]
        for j in range(0,len(test_data.daysets)):
            groups = list(test_data.daysets)
            test_grp = groups.pop(j)
            # test_saved(np=train_all.non_prob, test_grp=test_grp, params=params)

            test = TestModel(
                H_num=h, 
                model=train_all.model, 
                non_prob=train_all.non_prob, 
                test_data=test_grp,
                lag=params['lag'],
                min_inc=params['min_inc'],
                lag_type=params['lag_type'],
                )

            test.predictions.to_csv(f'/Users/maggie/Desktop/final_predictions/{h}_grp{j}.csv')
            test.metrics['test home'] = h
            test.metrics['self/cross'] = 'cross'

            metrics.append(test.metrics)    
    sys.exit()        

    metrics_df = pd.concat(metrics)
    # coeffs_df = pd.DataFrame(coeffs)
    return coeffs, metrics_df



# def train_all_combined(all_homes, params):
#     train_data = []


#     train_df = pd.concat(train_data)
#     sm = train_statsmodel(df=train_df)
#     print(sm)
#     sys.exit()
    train_all = TrainModel(
        H_num=f'all',
        train_data=train_df, 
        C=params['C'],
        penalty=params['penalty'],
        lag=params['lag']
        )
    # coeffs[f'train all'] = train_all.coeffs
    # coeffs = pd.DataFrame(train_all.coeffs)
    # train_all.coeffs.to_csv('/Users/maggie/Desktop/coeffs_all_combined.csv')
    # 

def load_model(model_to_load):    
    print('loading saved model....')
    with open(model_to_load, 'rb') as model_file:
        model = pickle.load(model_file)  
    
    print('loaded coefs:')
    print(model.coef_)
    return model

def test_saved(np, test_grp, params):

    new_model_loc = '/Users/maggie/Desktop/ALRL_models/new_model1.pickle'
    model = load_model(new_model_loc)

    print('testing loaded model... ')
    test = TestModel(
        H_num=f'all', 
        model=model, 
        non_prob=np, 
        test_data=test_grp,
        lag=params['lag'],
        min_inc=params['min_inc'],
        lag_type=params['lag_type'],
        )
    print(test.metrics)
    sys.exit()
    



# def test_combined(all_homes, params):
#     metrics = []
#     coeffs = {}
#     # print('allllll homes', all_homes)

#     for h in sorted(all_homes, reverse=True):
#         train_homes = [x for x in all_homes if x != h]
#         train_data = [all_homes[x].df for x in train_homes]
#         train_df = pd.concat(train_data)
#         test_data = all_homes[h]

#         train_all = TrainModel(
#             H_num=f'~{h}', 
#             train_data=train_df, 
#             C=params['C'],
#             penalty=params['penalty'],
#             lag=params['lag']
#             )

#         coeffs[f'test {h} all'] = train_all.coeffs
#         for j in range(0,len(test_data.daysets)):
#             groups = list(test_data.daysets)
#             test_grp = groups.pop(j)
#             # test_saved(np=train_all.non_prob, test_grp=test_grp, params=params)

#             test = TestModel(
#                 H_num=f'~{h}', 
#                 model=train_all.model, 
#                 non_prob=train_all.non_prob, 
#                 test_data=test_grp,
#                 lag=params['lag'],
#                 min_inc=params['min_inc'],
#                 lag_type=params['lag_type'],
#                 )

#             test.metrics['test home'] = h
#             test.metrics['self/cross'] = 'cross'
#             metrics.append(test.metrics)
#             print(test.metrics)
#             # print('exiting...')
#             # sys.exit()
        
#         train_single = all_homes[h]
#         for i in range(0,len(train_single.daysets)):
#             self_groups = list(train_single.daysets)
#             test_self = self_groups.pop(i)
#             train_self = pd.concat(self_groups)

#             single = TrainModel(
#                 H_num=h, 
#                 train_data=train_self, 
#                 C=params['C'],
#                 penalty=params['penalty'],
#                 lag=params['lag']
#                 )

#             coeffs[f'test {h} {i}'] = single.coeffs

#             self_test = TestModel(
#                 H_num=h, 
#                 model=single.model, 
#                 non_prob=single.non_prob, 
#                 test_data=test_self,
#                 lag=params['lag'],
#                 min_inc=params['min_inc'],
#                 lag_type=params['lag_type']
#                 )
#             self_test.metrics['test home'] = h
#             self_test.metrics['self/cross'] = 'self'
#             metrics.append(self_test.metrics)
            
#     metrics_df = pd.concat(metrics)
#     coeffs_df = pd.DataFrame(coeffs)
#     return coeffs_df, metrics_df



def get_classifier_summaries(df):
    print(df)
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
# homes = ['H1', 'H2'] 





# p = 4

# params_to_change = [
#     ('lag_type', ['avg', 'static']), 
#     ('min_inc', [5,15,30,60]),
#     ('grp_len', [11,8,5]),
#     ('C', [0.07, 0.1, 0.3]),
#     ('penalty', ['l1', 'l2'])
#     ]

# change_item = 'train_len'
fname = 'saved_final'
store_dir = f'/Users/maggie/Desktop/{fname}'
print(store_dir)
os.makedirs(store_dir, exist_ok=True)
parameters_set = param_dict



# # change_item = params_to_change[p][0]
# # change_values = params_to_change[p][1]

# # store_dir = f'/Users/maggie/Desktop/all_results/change_{change_item}'
# # print(store_dir)
# # os.makedirs(store_dir, exist_ok=True)
# # # sys.exit()

# # combined_param_results = []
# # summary_param_results = []
# # for x in change_values:

# parameters_set = set_params(change_item=change_item, value=x)
# fname = get_fname(parameters_set)
# print(parameters_set)
home_data = generate_datasets(homes, params=parameters_set)

# full_coeffs, full_metrics = train_all_combined(all_homes=home_data, params=parameters_set)
# full_coeffs, full_metrics = train_balanced(all_homes=home_data, params=parameters_set)
full_coeffs, full_metrics= get_balanced_df(all_homes=home_data, params=parameters_set, fname=fname)
# sys.exit()

# full_coeffs, full_metrics = test_combined(all_homes=home_data, params=parameters_set)
full_coeffs.to_csv(os.path.join(store_dir, f'coeffs_{fname}.csv'))
full_metrics.to_csv(os.path.join(store_dir, f'metrics_{fname}.csv'))
print(full_metrics)

full, by_home = get_classifier_summaries(full_metrics)

by_home.to_csv(os.path.join(store_dir, f'home_gpby_{fname}.csv'))
full.to_csv(os.path.join(store_dir, f'full_gpby_{fname}.csv'))
full['test home'] = 'H0'

cols = ['self/cross', 'test home', 'Accuracy', 'F1', 'F1 neg', 'Accuracy std']

group_results = full.loc[['AR predictions','NP'], cols]
home_results = by_home.loc[['AR predictions','NP'], cols]

joined_results = pd.concat([group_results, home_results])
joined_results = attach_params(df=joined_results, params=parameters_set)
# combined_param_results.append(joined_results)
    
# final_df = pd.concat(combined_param_results)
# final_df.to_csv(f'~/Desktop/all_results/_final_{change_item}.csv')
joined_results.to_csv(os.path.join(store_dir,f'_{fname}.csv'))