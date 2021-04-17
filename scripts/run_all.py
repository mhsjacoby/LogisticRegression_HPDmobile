"""
run_all.py
Authors: Maggie Jacoby
Last update: 2021-04-15
"""


import os
import sys
import argparse
import pandas as pd
from glob import glob
from datetime import datetime, date

from train import TrainModel
from test import TestModel

local_save_path = '/Users/maggie/Desktop'
parent_dir = os.path.dirname(os.getcwd())




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
        hub='RS4'
        )

# H2 = TrainModel(
#         H_num='H2',
#         cv=args.cv,
#         fill_type=args.fill_type,
#         # hub='RS2'
#         )

# H3 = TrainModel(
#         H_num='H3',
#         cv=args.cv,
#         fill_type=args.fill_type,
#         # hub='RS3'
#         )

# H4 = TrainModel(
#         H_num='H4',
#         cv=args.cv,
#         fill_type=args.fill_type,
#         )

# H5 = TrainModel(
#         H_num='H5',
#         cv=args.cv,
#         fill_type=args.fill_type,
#         )

# H6 = TrainModel(
#         H_num='H6',
#         cv=args.cv,
#         fill_type=args.fill_type,
#         )


# homes = [H1, H2, H3, H4, H5, H6]
homes = [H1]


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
        
        T.predictions.to_csv(os.path.join(parent_dir, 'Results', train.H_num, f'{test.H_num}_predictions.csv'), index=True)

        metric_df = T.metrics
        metric_df.index.name = 'function'
        metric_df['train'] = train.H_num
        metric_df['test'] = test.H_num

        metric_df['self/cross'] = 'self' if train.H_num == test.H_num else 'cross'
        
        metric_df.set_index(metric_df.index, inplace=True)

        all_metrics.append(metric_df)


df = pd.concat(all_metrics)
df = df.sort_values(by=['function', 'self/cross'], ascending=[True, False])

df.to_csv(os.path.join(parent_dir, 'Results', f'{fname}_metrics.csv'), index=True)

coeffs_df = pd.DataFrame(coeff_list)
coeffs_df.to_csv(os.path.join(parent_dir, 'Results', f'{fname}_coeffs.csv'), index=True)

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
