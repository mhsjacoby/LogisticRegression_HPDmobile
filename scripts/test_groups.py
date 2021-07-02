"""
test_groups.py
Authors: Maggie Jacoby
Last update: 2021-06-26
"""


import os
import sys
import argparse
import pandas as pd
from glob import glob
from datetime import datetime, date

from etl import ETL
from train import TrainModel
from test import TestModel

local_save_path = '/Users/maggie/Desktop'
parent_dir = os.getcwd()


homes = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
# homes = ['H1', 'H4']#, 'H6']


def test_model(tr, ts, trained_model, test_data, c):
    c += 1

    test = TestModel(
                    H_num=tr,
                    model=trained_model.model, 
                    non_param=trained_model.non_parametric_model,
                    test_data=test_data
                    )

    test.metrics['train'] = tr
    test.metrics['test'] = ts
    test.metrics['train Len'] = len(trained_model.X)
    test.metrics['test Len'] = len(test_data)
    test.metrics['self/cross'] = 'self' if tr == ts else 'cross'

    return test.metrics, c

# coeffs = {}
all_homes = {}
for h in homes:
    x = ETL(H_num=h)
    x.generate_dataset()
    all_homes[h] = x
    print(h, 'num sets', len(x.daysets))

all_metrics = []
c = 0

for train_h in all_homes:
    train_data = all_homes[train_h]

    ## train subgroup models
    for i in range(0,len(train_data.daysets)):
        groups = list(train_data.daysets)
        test_slf = groups.pop(i)
        train = pd.concat(groups)

        train_grp = TrainModel(H_num=train_h, train_data=train)
        print('*****', train_h, 'self', len(train), len(test_slf))

        ### test on self subgroups
        metrics, c = test_model(tr=train_h, ts=train_h, trained_model=train_grp, test_data=test_slf, c=c)
        metrics['train type'] = f'{train_h} grp~{i}'
        metrics['test type'] = f'{train_h} grp {i}'

        all_metrics.append(metrics)

        for test_h in all_homes:
            test_data = all_homes[test_h]
            if test_h == train_h:
                # print(f'skipping {train_h} group - {test_h} all')
                continue
            ## for each grp trained model, test on cross full
            # metrics, c = test_model(tr=train_h, ts=test_h, trained_model=train_grp, test_data=test_data.df, c=c)
            # # metrics['name'] = f'groups ~{i} train, cross full test'
            # metrics['train type'] = f'groups ~{i}'
            # metrics['test type'] = f'full'
            # all_metrics.append(metrics)

            ## for each grp trained model, test on cross groups
            for j in range(0,len(test_data.daysets)):
                groups = list(test_data.daysets)
                test_crssGrp = groups.pop(j)
                # c+=1
                print('*****', train_h, i, test_h, j, len(train), len(test_crssGrp))

                metrics, c = test_model(tr=train_h, ts=test_h, trained_model=train_grp, test_data=test_crssGrp, c=c)
                # metrics['name'] = f'group ~{i} train, cross group {j} test'
                metrics['train type'] = f'{train_h} grp~{i}'
                metrics['test type'] = f'{test_h} grp {j}'
                all_metrics.append(metrics)

    ## train model on full dataset
    train_all = TrainModel(H_num=train_h, train_data=train_data.df)
    for test_h in all_homes:
        test_data = all_homes[test_h]
        if test_h == train_h:
            # print(f'skipping {train_h} all - {test_h} all')
            continue
        # metrics, c = test_model(tr=train_h, ts=test_h, trained_model=train_all, test_data=test_data.df, c=c)
        # # metrics['name'] = 'full train, cross full test'
        # metrics['train type'] = f'full'
        # metrics['test type'] = f'full'
        # all_metrics.append(metrics)

        ## for each full trained model, test on cross groups
        for k in range(0,len(test_data.daysets)):
            groups = list(test_data.daysets)
            test_crssGrp = groups.pop(k)
            print('*****', train_h, test_h, k, len(train), len(test_crssGrp))


            metrics, c = test_model(tr=train_h, ts=test_h, trained_model=train_all, test_data=test_crssGrp, c=c)
            # metrics['name'] = f'full train, cross group {k} test'
            metrics['train type'] = f'{train_h} full'
            metrics['test type'] = f'{test_h} grp {k}'
            all_metrics.append(metrics)


print(f'******** total models tested: {c} ********')
all_metrics_df = pd.concat(all_metrics)
all_metrics_df.to_csv('~/Desktop/combined_metrics_allHomes_discard_day1.csv')
