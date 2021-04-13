"""
prediction_functions.py
Author: Maggie Jacoby
Date: 2021-04-13
"""

# import os
# import sys
# import csv
# import yaml
# import logging
# import argparse
import numpy as np
import pandas as pd
# from glob import glob
from datetime import datetime, date
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, f1_score, accuracy_score, matthews_corrcoef


def baseline_OR(X, y, thresh=0.5):
    """Get baseline results to compare LR to

    Uses the previous OR gate and generate predictions

    Returns: y_hat (predictions) and f1(rev) and accuracy
    """
    base_cols = ['audio', 'img']
    full_cols =  base_cols + ['temp', 'rh', 'co2eq', 'light']

    for cols, title in zip([base_cols, full_cols], ('AI', 'AIE')):
        df = X[cols].copy()
        pred = str('y_hat_' + title)
        df[pred] = 0 
        df.loc[df.max(axis=1) > thresh, pred] = 1
        y_hat = df[pred].to_numpy()
        a_, b_, blm = get_model_metrics(y_true=y, y_hat=y_hat)
        # self.metrics.update({title+' F1 neg': blm['F1 neg'], title+' F1': blm['F1'], title + ' Acc': blm['Accuracy']})
        print('=== pred !!', pred)
        print('=== a_ !!', a_)
        print('=== b_ !!', b_)
        print('=== blm !!', blm)
        print('==========')







def get_predictions_wGT(logit_clf, X_df):
    """Run data through classifier to get predictions given X and y using ground truth for lags

    Returns: probabilities (between 0,1) and predictions (0/1) as a df
    """
    # print(X_df)
    X = X_df.to_numpy()

    probs = logit_clf.predict_proba(X)[:,1]
    preds = logit_clf.predict(X)
    df = pd.DataFrame(
                    data=np.transpose([probs, preds]), 
                    index=X_df.index,
                    columns=['Probability', 'Predictions']
                    )
    # print(df)
    return df


def get_predictions_nonparametric():
    """Create likilihood of occupancy, based only on past occupancy
    """
    pass



def test_with_predictions(logit_clf, X, hr_lag=8):
    """Run data through classifier and push predictions forward as lag values

    This is used instead of get_predictions_wGT.
    Returns: probabilities (between 0,1) and predictions (0/1) as a df

    """

    lag_max = hr_lag*12

    X_start = X.iloc[:lag_max]
    lag_cols=[c for c in X.columns if c.startswith('lag')]
    exog_vars = X.drop(columns=lag_cols).iloc[lag_max:]
    preds_X = pd.concat([X_start, exog_vars])
    preds_X.index = pd.to_datetime(preds_X.index)

    ys = []

    for idx, _ in preds_X.iterrows():
        df_row = preds_X.loc[idx]
        curr_row = df_row.to_numpy().reshape(1,-1)

        y_hat = logit_clf.predict(curr_row)
        y_proba = logit_clf.predict_proba(curr_row)[:,1]
        idx_loc = preds_X.index.get_loc(idx)

        for j in range(1, hr_lag + 1):
            lag_col_name = f'lag{j}_occupied'
            ind_to_set = idx_loc + j*12
            try:
                preds_X.at[preds_X.iloc[ind_to_set].name, lag_col_name] = y_hat[0]
            except:
                continue

        ys.append((idx, y_proba[0], y_hat[0]))
    y_hats = pd.DataFrame(ys).set_index(0)
    y_hats.index.name = 'timestamp'
    y_hats.columns = ['Probability', 'Predictions']

    return y_hats