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

# ks = [i for i in range(3,13)]
# print(ks)

coeff_list = []

# ks = [3,7]
# homes = ['H1', 'H2']
# best_cs = []
Cs = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
# for k in ks:
all_data = []
# Cs = {}

for h in homes:
    x = ETL(H_num=h)
    x.generate_dataset()
    all_data.append(x.df)
    # h_coefs = {}
    for c in Cs:
        print('================', c)
        # print(f'=========== k:{k}, {h}')

        cv = TrainModel(H_num=h, cv=False, train_data=x.df, C=c)
        # clf = cv.train_model()
        # print(h, cv.C)
        # coeff_list[h] = cv.coeffs
        coeff_list.append(cv.coeffs)
        # print(cv.coeffs)
        # sys.exit()
        # dct = dict(home=h, C=cv.C, score=cv.train_score)
        # best_cs.append(dct)
    # Cs[h] = cv.C

# print(f'=========== k:{k}, all')
all_dfs = pd.concat(all_data)
for c in Cs:
    # print(f'=========== k:{k}, {h}')

    all_data_cv = TrainModel(H_num='all', cv=False, train_data=all_dfs, C=c)
            # clf = cv.train_model()
    # print(h, cv.C)
    # coeff_list[h] = cv.coeffs
    coeff_list.append(all_data_cv.coeffs)


# all_data_cv = TrainModel(H_num=h, cv=True, train_data=all_dfs)
# clf = all_data_cv.train_model(k=k)
# print('All best C:', all_data_cv.C)
# dct = dict(home='all', C=cv.C, score=cv.train_score)
# coeff_list['all'] = all_data_cv.coeffs
# best_cs.append(dct)

# all_cs = pd.DataFrame(best_cs)
# print(all_cs)
# all_cs.to_csv('~/Desktop/All_best_01C.csv')
coeff_df = pd.DataFrame(coeff_list)
coeff_df.to_csv('~/Desktop/full_coeffs_setC.csv')