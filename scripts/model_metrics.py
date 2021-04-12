"""
model_metrics.py
Author: Maggie Jacoby
Date: 2021-04-12
"""

import os
import sys
import csv
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, date
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, f1_score, accuracy_score, matthews_corrcoef


def get_predictions_wGT(logit_clf, X_df):
    """Run data through classifier to get predictions given X and y using ground truth for lags

    Returns: probabilities (between 0,1) and predictions (0/1) as a df
    """

    X = X_df.to_numpy()

    probs = logit_clf.predict_proba(X)[:,1]
    preds = logit_clf.predict(X)
    df = pd.DataFrame(
                    data=np.transpose([probs, preds]), 
                    index=X_df.index,
                    columns=['Probability', 'Predictions']
                    )
    return df


def get_predictions_nonparametric():
    """Create likilihood of occupancy, based only on past occupancy
    """
    pass




def get_model_metrics(y_true, y_hat):
    """Stand-alone function to get metrics given a classifier.

    Returns: Confusion matrix and list of results as tuples
    """
    conf_mat = pd.DataFrame(confusion_matrix(y_true, y_hat), 
                            columns = ['Vacant', 'Occupied'],
                            index = ['Vacant', 'Occupied']
                            )
    conf_mat = pd.concat([conf_mat], keys=['Actual'], axis=0)
    conf_mat = pd.concat([conf_mat], keys=['Predicted'], axis=1)

    score = accuracy_score(y_true, y_hat)
    RMSE = np.sqrt(mean_squared_error(y_true, y_hat))
    f1 = f1_score(y_true, y_hat, pos_label=1)
    f1_rev = f1_score(y_true, y_hat, pos_label=0)

    # mcc = matthews_corrcoef(y_true, y_hat)

    results_metrics = [
                        ('length', len(y_true)),
                        ('Accuracy', f'{score:.4}'),
                        ('RMSE', f'{RMSE:.4}'),
                        ('F1', f'{f1:.4}'),
                        ('F1 neg', f'{f1_rev:.4}'),
                        # ('MCC', f'{mcc:.4}')
                        ]
    
    # metrics = [r[1] for r in results_metrics[1:]]
    metrics = {r[0]: r[1] for r in results_metrics}

    return conf_mat, results_metrics, metrics


def additional_metrics(conf_mat):

    tn, fp, fn, tp = conf_mat.ravel()
    print(f'\ntn: {tn} fp:{fp} fn:{fn}, tp:{tp}')

    tpr = tp/(tp+fn) if tp+fn > 0 else 0.0
    fpr = fp/(tn+fp) if tn+fp > 0 else 0.0

    tnr = tn/(tn+fp) if tn+fp > 0 else 0.0
    fnr = fn/(tp+fn) if tp+fn > 0 else 0.0

    return {'tnr': f'{tnr:.3}', 'fpr': f'{fpr:.3}', 'fnr': f'{fnr:.3}', 'tpr': f'{tpr:.3}'}
