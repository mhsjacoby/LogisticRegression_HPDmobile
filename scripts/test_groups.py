"""
test_groups.py
Authors: Maggie Jacoby
Last update: 2021-06-26
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

local_save_path = '/Users/maggie/Desktop'
parent_dir = os.getcwd()


def test_model(tr, ts, trained_model, test_data, lag):

    test = TestModel(
                    H_num=tr,
                    model=trained_model.model, 
                    non_param=trained_model.non_parametric_model,
                    test_data=test_data,
                    lag=lag
                    )

    test.metrics['train'] = tr
    test.metrics['test'] = ts
    test.metrics['train Len'] = len(trained_model.X)
    test.metrics['self/cross'] = 'self' if tr == ts else 'cross'

    return test.metrics


def cross_test(model, metrics):
    train_h = model.H_num
    grp_acc, grp_mcc = [], []
    grp_f1, grp_f1N = [], []
    for test_h in all_homes:
        test_data = all_homes[test_h]
        if test_h == train_h:
            continue

        ## for each subgroup trained model, cross test on all other homes' subgroups
        for j in range(0,len(test_data.daysets)):
            groups = list(test_data.daysets)
            test_grp = groups.pop(j)
            test_metrics = test_model(tr=train_h, ts=test_h, trained_model=model, test_data=test_grp)
            test_metrics['test group'] = j
            metrics.append(test_metrics)
            grp_acc.append(np.float(test_metrics.loc['AR predictions']['Accuracy']))
            grp_mcc.append(np.float(test_metrics.loc['AR predictions']['MCC']))
            grp_f1.append(np.float(test_metrics.loc['AR predictions']['F1']))
            grp_f1N.append(np.float(test_metrics.loc['AR predictions']['F1 neg']))
    return metrics, grp_acc, grp_mcc, grp_f1, grp_f1N



def test_combined(all_homes, c, metrics, coeffs, lag):
    
    for h in sorted(all_homes, reverse=True):
        train_homes = [x for x in all_homes if x != h]
        train_data = [all_homes[x].df for x in train_homes]
        train_df = pd.concat(train_data)
        train_df = train_df.dropna()
        test_data = all_homes[h]
        train_all = TrainModel(H_num='H0', train_data=train_df, cv=False, C=c, lag=lag)
        grp_acc, grp_mcc = [], []
        grp_f1, grp_f1_neg = [], []

        for j in range(0,len(test_data.daysets)):
            groups = list(test_data.daysets)
            test_grp = groups.pop(j)
            nans = test_grp.isnull().sum()
            if nans.sum() > 0:
                print(f'{nans.sum()} in group {j}, home {h}')
                continue
            test_metrics = test_model(tr='H0', ts=h, trained_model=train_all, test_data=test_grp, lag=lag)
            test_metrics['C'] = c
            test_metrics['test group'] = j
            metrics.append(test_metrics)
            
            grp_acc.append(np.float(test_metrics.loc['AR predictions']['Accuracy']))
            grp_mcc.append(np.float(test_metrics.loc['AR predictions']['MCC']))
            grp_f1.append(np.float(test_metrics.loc['AR predictions']['F1']))
            grp_f1_neg.append(np.float(test_metrics.loc['AR predictions']['F1 neg']))

        train_all.coeffs['cross acc'] = np.mean(grp_acc)
        train_all.coeffs['cross mcc'] = np.mean(grp_mcc)
        train_all.coeffs['cross f1'] = np.mean(grp_f1)
        train_all.coeffs['cross f1 neg'] = np.mean(grp_f1_neg)
        coeffs[f'test {h}, {c}'] = train_all.coeffs
    
    return coeffs, metrics


if __name__=='__main__':
    homes = ['H1', 'H2', 'H3', 'H5', 'H6']
    # homes = ['H6', 'H5']

    # Cs = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9]
    Cs = [0.03, 0.1, 0.3]
    lag = 8
    DFs = []
    all_metrics = []
    full_metrics = []
    full_coeffs = {}

    all_homes = {}
    ## Generate all datasets for all homes
    for h in homes:
        # if h == 'H4':
        #     continue
        x = ETL(H_num=h, lag=lag)
        x.generate_dataset()
        all_homes[h] = x
        print(h, 'num sets', len(x.daysets))


    for c in Cs:
        full_coeffs, full_metrics = test_combined(all_homes=all_homes, c=c, metrics=full_metrics, coeffs=full_coeffs, lag=lag)

    #     all_coeffs  = {}
    #     for train_h in all_homes:
    #         train_data = all_homes[train_h]

    #         ## train subgroup models
    #         for i in range(0,len(train_data.daysets)):
    #             groups = list(train_data.daysets)
    #             test_slf = groups.pop(i)
    #             train = pd.concat(groups)

    #             train_grp = TrainModel(H_num=train_h, train_data=train, cv=False, C=c) 

    #             ### test on self subgroups
    #             metrics = test_model(tr=train_h, ts=train_h, trained_model=train_grp, test_data=test_slf)
    #             metrics['test group'] = i
                
    #             train_grp.coeffs['self acc'] = metrics.loc['AR predictions']['Accuracy']
    #             train_grp.coeffs['self mcc'] = metrics.loc['AR predictions']['MCC']
    #             train_grp.coeffs['self f1'] = metrics.loc['AR predictions']['F1']
    #             train_grp.coeffs['self f1 neg'] = metrics.loc['AR predictions']['F1 neg']

    #             train_grp.coeffs['gt acc'] = metrics.loc['AR ground truth']['Accuracy']
    #             train_grp.coeffs['gt mcc'] = metrics.loc['AR ground truth']['MCC']
    #             train_grp.coeffs['gt f1'] = metrics.loc['AR ground truth']['F1']
    #             train_grp.coeffs['gt acc'] = metrics.loc['AR ground truth']['F1 neg']


    #             all_metrics.append(metrics)

    #             all_metrics, grp_acc, grp_mcc, grp_f1, grp_f1N = cross_test(model=train_grp, metrics=all_metrics)
    #             train_grp.coeffs['cross acc'] = np.mean(grp_acc)
    #             train_grp.coeffs['cross mcc'] = np.mean(grp_mcc)
    #             train_grp.coeffs['cross f1'] = np.mean(grp_f1)
    #             train_grp.coeffs['cross f1 neg'] = np.mean(grp_f1N)
    #             all_coeffs[f'{train_h}-{i}, {c}'] = train_grp.coeffs

    #         # train model on full dataset
    #         train_full = TrainModel(H_num=train_h, train_data=train_data.df, cv=False, C=c)
    #         all_metrics, full_acc, full_mcc, full_f1, full_f1N = cross_test(model=train_full, metrics=all_metrics)

    #         train_full.coeffs['cross acc'] = np.mean(full_acc)
    #         train_full.coeffs['cross mcc'] = np.mean(full_mcc)
    #         train_full.coeffs['cross f1'] = np.mean(full_f1)
    #         train_full.coeffs['cross f1 neg'] = np.mean(full_f1N)
    #         all_coeffs[f'{train_h}-full, {c}'] = train_full.coeffs

    #     coeffs_df = pd.DataFrame(all_coeffs)
    #     DFs.append(coeffs_df)

    #     print(f'******** C ={c}********')
    #     all_metrics_df = pd.concat(all_metrics)
    #     all_metrics_df['C'] = c
    #     all_metrics_df.to_csv(f'~/Desktop/all_metrics_{c}.csv')


    # coeffs = pd.concat(DFs, axis=1)
    # coeffs.to_csv('~/Desktop/all_CV_coeffs.csv')


    ## for metrics representing joined homes 
    full_coeffs_df = pd.DataFrame(full_coeffs)
    full_coeffs_df.to_csv(f'~/Desktop/joined_homes_CV_coeffs_lag{lag}_3c.csv')

    full_metrics_df = pd.concat(full_metrics)
    full_metrics_df.to_csv(f'~/Desktop/joined_homes_CV_metrics_lag{lag}_3c.csv')
