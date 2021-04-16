"""
model_metrics.py
Author: Maggie Jacoby
Date: 2021-04-13
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import f1_score, accuracy_score


def get_model_metrics(y_true, y_hat):
    """Stand-alone function to get metrics given classifier results.

    Returns: confusion matrix and dictionary with results
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

    results_metrics = {
                        'Accuracy': f'{score:.4}',
                        'RMSE': f'{RMSE:.4}',
                        'F1': f'{f1:.4}',
                        'F1 neg': f'{f1_rev:.4}',
                        }

    results_metrics.update(counts(confusion_matrix(y_true, y_hat)))
    return conf_mat, results_metrics



def counts(conf_mat):

    tn, fp, fn, tp = conf_mat.ravel()
    # print(f'\ntn: {tn} fp:{fp} fn:{fn}, tp:{tp}')

    tpr = tp/(tp+fn) if tp+fn > 0 else 0.0
    fpr = fp/(tn+fp) if tn+fp > 0 else 0.0
    tnr = tn/(tn+fp) if tn+fp > 0 else 0.0
    fnr = fn/(tp+fn) if tp+fn > 0 else 0.0
    
    all_counts = {
                    'tnr': f'{tnr:.3}',
                    'fpr': f'{fpr:.3}',
                    'fnr': f'{fnr:.3}', 
                    'tpr': f'{tpr:.3}',
                    'tn': tn,
                    'fp': fp,
                    'fn': fn,
                    'tp': tp
                    }

    return all_counts