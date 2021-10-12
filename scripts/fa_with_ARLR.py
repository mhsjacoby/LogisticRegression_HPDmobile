"""
fa_with_ARLR.py
Author: Maggie Jacoby
Date: 2021-07-28
"""

import os
import sys
import csv
import json
import argparse
import itertools
import pickle
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timedelta, time

from functools import reduce
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from etl import ETL
from test_model import TestModel


def load_model(model_to_load=''):
    """ loads an already trained sklearn logit model
    """
    print(f'>>> loading saved model.... {model_to_load}')
    with open(model_to_load, 'rb') as model_file:
        model = pickle.load(model_file)  
    print('>>> loaded coefs:')
    print(model.coef_)
    return model


def get_hubs(num_hubs, level='full'):
    """ returns a list of hubs to cycle throu
    """
    spec_path = '/Users/maggie/Documents/Github/LogisticRegression_HPDmobile/FactorialAnalysis/fracfact_output'
    run_file = os.path.join(spec_path, f'{num_hubs}_hub_{level}.csv')

    run_specifications = []
    with open(run_file) as FFA_output:
        for i, row in enumerate(csv.reader(FFA_output), 1):
            run_specifications.append((i, row))
    return run_specifications


def runVS(hub_list, run_spec):
    """ returns a run where some hubs are fully used, some not used at all
    """
    hubs_to_use = []
    for i, hub in zip(run_spec, hub_list):
        if i == '1':
            hubs_to_use.append(hub)
    return hubs_to_use


def get_null(df):
    cols_to_change = [ 'audio', 'co2eq', 'light', 'rh', 'temp', 'img']
    for col in cols_to_change:
        df[col] = 0.0
    return df



def run_tests(model, test_data):
    """ given some model and a test dataset, runs the test and return the results
    """
    test_results = []
    for test_grp in test_data.daysets:
        test = TestModel(test_data=test_grp)
        metrics = test.test_single(clf=model)
        test_results.append(metrics)
    results_df = pd.DataFrame(test_results)
    results_df = results_df.apply(pd.to_numeric)

    var = results_df.var(axis=0)
    results_df = results_df.mean(axis=0)
    
    return results_df, var

# comp_type = 'best-env'
# comp_type= 'env'
ffa_storage = '/Users/maggie/Desktop/FA_withARLR_results/completed_runs'
homes = {'H1':5, 'H2':4, 'H3':5, 'H5':5, 'H6':4}
# homes = {'H6':4}

model_loc = '/Users/maggie/Documents/Github/LogisticRegression_HPDmobile/FactorialAnalysis/'
model_to_load = os.path.join(model_loc, 'ARLR_model.pickle')
model = load_model(model_to_load)

comparisons = ['audio-img', 'best-env', 'hub']
for comp_type in comparisons:
    for home, hubs in homes.items():
        run_names = []
        TestHome = ETL(H_num=home)
        full_run = []
        variance = []
        run_specs = get_hubs(num_hubs=hubs)
        for sp in run_specs:
            print(f'=====!!!!!!!!!!!!!!!!==== RUN {sp} =====!!!!!!!!!!!!!!!!==== ')
            run_names.append(f'run {sp[0]}')
            hubs = runVS(hub_list=TestHome.all_hubs, run_spec=sp[1])
            TestHome.generate_dataset(hub=hubs, mod=comp_type)

            if len(hubs) == 0 and comp_type == 'hub':
                TestHome.df = get_null(TestHome.df)
            
            results, var = run_tests(test_data=TestHome, model=model)
            variance.append(var)
            df_incl = pd.DataFrame(sp[1], index=TestHome.all_hubs)
            df = pd.concat([df_incl, results], axis=0)
            full_run.append(df)
        variance_df = pd.DataFrame(variance)
        var_estimate = variance_df.mean(axis=0)
        full_run.append(var_estimate)
        run_names.append(f'run var')
        all_runs = pd.concat(full_run, axis=1)
        all_runs = all_runs.set_axis(run_names, axis=1)
        all_runs = all_runs.T
        all_runs.index.name = 'Run'

        all_runs.to_csv(os.path.join(ffa_storage, f'{home}_{comp_type}.csv'))

    # sys.exit()


