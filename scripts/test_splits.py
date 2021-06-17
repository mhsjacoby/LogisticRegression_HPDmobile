"""
test_splits.py
Authors: Maggie Jacoby
Last update: 2021-06-04
"""


import os
import sys
import argparse
import pandas as pd
from glob import glob
from datetime import datetime, date

from etl import ETL
from trainAR import TrainModel
from testAR import TestModel

local_save_path = '/Users/maggie/Desktop'
# parent_dir = os.path.dirname(os.getcwd())
parent_dir = os.getcwd()




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train and test all models')
    parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')
    parser.add_argument('-cv', '--cv', default=False, action='store_true', help='Perform cross-validation?')
    parser.add_argument('-fname', '--fname', default='', type=str, help='name to append to saved files')
    args = parser.parse_args()

    fname = args.fname

    if not args.cv:
        args.cv = False


# H1 = TrainModel(
#         H_num='H1',
#         cv=args.cv,
#         fill_type=args.fill_type,
#         # hub='RS4'
#         )

splits = [round(i*.1,1) for i in range(2,9,1)]
# all_home_objs = {}
all_metrics = []


homes = ['H1', 'H2', 'H3', 'H5', 'H6']
for home in homes:
    for s in splits:

        data = ETL(H_num=home, split=s)
        data.generate_dataset()
        train1 = TrainModel(
                        H_num=home,
                        train_data=data.train,
                        )
        test1 = TestModel(
                        H_num=home,
                        test_data=data.test,
                        model=train1.model,
                        non_param=train1.non_parametric_model,
                        )
        # print('first', data.split, len(train1.train), len(test1.test))
        metrics_df = test1.metrics
        metrics_df.index.name = 'function'
        metrics_df['split'] = s
        metrics_df['home'] = home
        metrics_df['direction'] = 'normal'
        metrics_df.set_index(metrics_df.index, inplace=True)
        all_metrics.append(metrics_df)

        train2 = TrainModel(
                        H_num=home,
                        train_data=data.test,
                        )
        test2 = TestModel(
                        H_num=home,
                        test_data=data.train,
                        model=train2.model,
                        non_param=train2.non_parametric_model,
                        )    
        # print('second', data.split, len(train2.train), len(test2.test))

        # print(data.split)
        # print('normal side')
        # print(test1.metrics)
        # print('reverse')
        # print(test2.metrics)

        metrics_df = test2.metrics
        metrics_df.index.name = 'function'
        metrics_df['split'] = s
        metrics_df['home'] = home
        metrics_df['direction'] = 'reverse'
        metrics_df.set_index(metrics_df.index, inplace=True)

        all_metrics.append(metrics_df)

df = pd.concat(all_metrics)
df = df.sort_values(by=['home', 'function', 'split', 'direction'], ascending=[True, False, True, True])

df.to_csv(os.path.join(parent_dir, 'Results', f'allHomes_splits_compared.csv'), index=True)


#     test = TestModel(
#                     H_num='H1',
#                     model=train.model,
#                     test_data=train.test,
#                     non_param=train.non_parametric_model,
#                     )
#     print(train.split)
#     print(test.metrics)

# print(len(all_home_objs))
# sys.exit()


# all_metrics = []
# coeff_list = {t.H_num: t.coeffs for t in homes}
# train_params = {t.H_num: t.configs for t in homes}
# best_cs = {t.H_num: t.C for t in homes}
# all_hubs = {x.H_num: x.hubs_to_use for x in homes}

# for hs in all_home_objs:
#     T = TestModel(H_num=hs.H_num, model=hs.model, test_data=hs.test, non_param=hs.non_parametric_model)
    
#     print(hs.split)
#     print(T.results)

# for train in homes:

#     # train.main()
#     for test in homes:
#         print(f'Model trained on {train.H_num}, tested on data from {test.H_num}')
        

#         # print('*****************************************************')
#         # sys.exit()

#         T = TestModel(
#             H_num=test.H_num,
#             model=train.model,
#             test_data=test.test,
#             non_param=train.non_parametric_model
#             )

        # sys.exit()
        # print(T.results)
        
        # os.makedirs(train.results_csvs)
        # print('train results dir', train.results_csvs)
        # store_results = os.path.join(parent_dir, 'Results', train.H_num)
        # os.makedirs(store_results, exist_ok=True)
        # T.predictions.to_csv(os.path.join(store_results, f'{test.H_num}_predictions.csv'), index=True)
        # T.predictions.to_csv(os.path.join(parent_dir, 'Results', 'Training_Predictions', f'{test.H_num}_predictions.csv'), index=True)
        # continue


#     print(T.metrics)
#     # T.metrics.to_csv('/Users/maggie/Desktop/metrics_test.csv', index=True)


    # metric_df = T.metrics
    # metric_df.index.name = 'function'
#     metric_df['train'] = train.H_num
#     metric_df['test'] = test.H_num

#     metric_df['self/cross'] = 'self' if train.H_num == test.H_num else 'cross'
    
#     metric_df.set_index(metric_df.index, inplace=True)

#     all_metrics.append(metric_df)

# # sys.exit()
# df = pd.concat(all_metrics)
# df = df.sort_values(by=['function', 'self/cross'], ascending=[True, False])

# df.to_csv(os.path.join(parent_dir, 'Results', f'{fname}_metrics.csv'), index=True)

# coeffs_df = pd.DataFrame(coeff_list)
# coeffs_df.to_csv(os.path.join(parent_dir, 'Results', f'{fname}_coeffs.csv'), index=True)

# file_to_write = \
#         f'hubs: {all_hubs}\n' \
#         f'fill type: {args.fill_type}\n' \
#         f'model train parameters: {train_params}\n' \
#         f'cv: {args.cv}\n' \
#         f'best C: {best_cs}'
#         # f'homes: {[x.H_num for x in homes]}\n' \


# text_file = os.path.join(parent_dir, 'Results', f'{fname}_saveparams.txt')
# with open(text_file, 'w') as f:
#         f.write(file_to_write)
