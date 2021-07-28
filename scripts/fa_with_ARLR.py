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


def load_model(model_to_load=''):
    """ loads an already trained sklearn logit model
    """
    print(f'>>> loading saved model.... {model_to_load}')
    with open(model_to_load, 'rb') as model_file:
        model = pickle.load(model_file)  
    print('>>> loaded coefs:')
    print(model.coef_)
    return model

def create_test_dataset(hubs=[]):
    """ given some list of hubs and a home, creates dataset for testing
    """
    df = pd.DataFrame()
    return df


def get_hubs():
    """ returns a list of hubs to cycle throu
    """
    hub_list = []
    return hub_list


def run_test(model, test_df):
    """ given some model and a test dataset, runs the test and return the results
    """
    results_df = pd.DataFrame()
    return results_df


def save_results():
    pass

