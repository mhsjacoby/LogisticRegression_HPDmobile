"""
train_test2.py
Authors: Maggie Jacoby
Last update: 2021-04-15
"""


import os
import argparse
import pandas as pd
from glob import glob
from datetime import datetime, date

from train import TrainModel
from test import TestModel

local_save_path = '/Users/maggie/Desktop'



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train and test all models')
    parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')
    parser.add_argument('-cv', '--cv', default=False, action='store_true', help='Perform cross-validation?')
    parser.add_argument('-fname', '--fname', default='', type=str, help='name to append to saved files')
    args = parser.parse_args()

    fname = args.fname

    if not args.cv:
        args.cv = False


H1 = TrainModel(
        H_num='H1',
        cv=args.cv,
        fill_type=args.fill_type,
        hub='RS1'
        )

H2 = TrainModel(
        H_num='H2',
        cv=args.cv,
        fill_type=args.fill_type,
        hub='RS4'
        )

homes = [H1, H2]

all_metrics = []
coeff_list = {t.H_num: t.coeffs for t in homes}
train_params = {t.H_num: t.configs for t in homes}
best_cs = {t.H_num: t.C for t in homes}
all_hubs = {x.H_num: x.hubs_to_use for x in homes}

for train in homes:

    for test in homes:
        print(f'Model trained on {train.H_num}, tested on data from {test.H_num}')
        
        T = TestModel(
            H_num=test.H_num,
            model=train.model,
            test_data=test.test,
            non_param=train.non_parametric_model
            )
        

        metric_df = T.metrics
        metric_df.index.name = 'function'
        metric_df['train'] = train.H_num
        metric_df['test'] = test.H_num
        metric_df.set_index(['train', 'test', metric_df.index], inplace=True)
        all_metrics.append(metric_df)


df = pd.concat(all_metrics)
print(df)
df.to_csv(os.path.join(local_save_path, f'{fname}_metrics.csv'), index=True)

coeffs_df = pd.DataFrame(coeff_list)
coeffs_df.to_csv(os.path.join(local_save_path, f'{fname}_coeffs.csv'), index=True)

file_to_write = \
        f'hubs: {all_hubs}\n' \
        f'fill type: {args.fill_type}\n' \
        f'model train parameters: {train_params}\n' \
        f'cv: {args.cv}\n' \
        f'best C: {best_cs}'
        # f'homes: {[x.H_num for x in homes]}\n' \



text_file = os.path.join(local_save_path, f'{fname}_saveparams.txt')
with open(text_file, 'w') as f:
        f.write(file_to_write)
