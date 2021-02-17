"""
explore.py
Authors: Maggie Jacoby and Jasmine Garland
Last update: 2021-02-16
"""

import os
import sys
import csv
import yaml
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, date
# from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
from sklearn.model_selection import ParameterGrid 

# from data_basics import DataBasics
from train import TrainModel
from test import TestModel
from etl import ETL 



class ExploreModels(TrainModel, TestModel):
    """Explore hyperparameters for logistic regression models

    Train for only one home, but can test for multiple homes
    """

    def __init__(self):
        pass

    
    def set_LR_parameters(self):
        """Sets the model parameters as specified in the configuration file.

        Takes in lists for the parameters and uses a parameter grid.
        Returns: sklearn logistic regression model object (not fit)
        """
        parameter_grid = ParameterGrid(self.configs)

        for param in parameter_grid:
            clf = LogisticRegression(**param)

        return clf
