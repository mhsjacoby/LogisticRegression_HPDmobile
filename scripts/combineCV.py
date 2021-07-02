"""
combineCV.py
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
# from test import TestModel

local_save_path = '/Users/maggie/Desktop'
parent_dir = os.getcwd()


homes = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']

ks = [i for i in range(3,13)]
print(ks)

# ks = [3,7]
# homes = ['H1', 'H2']
best_cs = []

for k in ks:
    all_data = []
    Cs = {}
    for h in homes:
        # print(f'=========== k:{k}, {h}')
        x = ETL(H_num=h)
        x.generate_dataset()
        all_data.append(x.df)
        cv = TrainModel(H_num=h, cv=True, train_data=x.df)
        clf = cv.train_model(k=k)
        # print(h, cv.C)
        dct = dict(home=h, k=k, C=cv.C, score=cv.train_score)
        best_cs.append(dct)
        # Cs[h] = cv.C

    print(f'=========== k:{k}, all')
    all_dfs = pd.concat(all_data)
    all_data_cv = TrainModel(H_num=h, cv=True, train_data=all_dfs)
    clf = all_data_cv.train_model(k=k)
    print('All best C:', all_data_cv.C)
    dct = dict(home='all', k=k, C=cv.C, score=cv.train_score)
    best_cs.append(dct)

all_cs = pd.DataFrame(best_cs)
print(all_cs)
all_cs.to_csv('~/Desktop/All_best_Cs_12.csv')