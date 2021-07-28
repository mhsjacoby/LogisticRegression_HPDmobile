"""
model_metrics.py
Author: Maggie Jacoby
Date: 2021-04-13
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef



def get_missing(y_true, y_hat):
    curr_val = int(list(y_true)[0])
    missing = set([0,1]) - set(y_true)
    missing_val = int(list(missing)[0])
    all_vals = ['Vacant', 'Occupied']
    curr_state = all_vals[curr_val]
    missing_state = all_vals[missing_val]
    conf_mat = pd.DataFrame(confusion_matrix(y_true, y_hat), 
                            columns = [curr_state], index=[curr_state])
    conf_mat[missing_state] = 0
    conf_mat.loc[missing_state]=[0, 0]
    return conf_mat





def get_model_metrics(y_true, y_hat):
    """Stand-alone function to get metrics given classifier results.

    Returns: confusion matrix and dictionary with results
    """
    conf_mat = pd.DataFrame(confusion_matrix(y_true, y_hat, labels=[0,1]), 
                            columns = ['Vacant', 'Occupied'],
                            index = ['Vacant', 'Occupied']
                            )
    # except:
    #     conf_mat = get_missing(y_true, y_hat)

    conf_mat = pd.concat([conf_mat], keys=['Actual'], axis=0)
    conf_mat = pd.concat([conf_mat], keys=['Predicted'], axis=1)

    score = accuracy_score(y_true, y_hat)
    RMSE = np.sqrt(mean_squared_error(y_true, y_hat))
    f1 = f1_score(y_true, y_hat, pos_label=1)
    f1_rev = f1_score(y_true, y_hat, pos_label=0)
    mcc = matthews_corrcoef(y_true, y_hat)

    results_metrics = {
                        'Accuracy': f'{score:.4}',
                        'RMSE': f'{RMSE:.4}',
                        'F1': f'{f1:.4}',
                        'F1 neg': f'{f1_rev:.4}',
                        'MCC': f'{mcc:.4}'
                        }

    results_metrics.update(counts(confusion_matrix(y_true, y_hat, labels=[0,1])))
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