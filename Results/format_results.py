"""
format_results.py
Author: Maggie Jacoby
Date: 2021-04-19
"""

import os
import sys
from glob import glob
import pandas as pd

parent_dir = os.path.dirname(os.getcwd())


def create_test_train_table(result_file, result_type='AR predictions'):
    to_read = glob(os.path.join(parent_dir, result_file, '*_metrics.csv'))[0]
    metrics = pd.read_csv(to_read, index_col='function')
    metrics = metrics

    



