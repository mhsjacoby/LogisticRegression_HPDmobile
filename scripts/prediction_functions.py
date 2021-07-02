"""
prediction_functions.py
Author: Maggie Jacoby
Date: 2021-04-15
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, date
import model_metrics as my_metrics


def baseline_OR(X, y, metrics, thresh=0.5):
    """Get baseline results to compare LR to

    Returns: y_hat (predictions) and f1(rev) and accuracy
    """
    
    base_cols = ['audio', 'img']
    full_cols =  base_cols + ['temp', 'rh', 'light', 'co2eq']
    predictions_df = pd.DataFrame(index=X.index)

    for cols, title in zip([base_cols, full_cols], ('OR (ai)', 'OR (aie)')):
        df = X[cols].copy()
        df['prob'] = df.max(axis=1)
        df['pred'] = 0 
        df.loc[df['prob'] > thresh, 'pred'] = 1
        y_hat = df['pred'].to_numpy()
        _, blm = my_metrics.get_model_metrics(y_true=y, y_hat=y_hat)
        metrics[title] = blm

        predictions_df[f'prob {title.strip(" OR()")}'] = df['prob']
        predictions_df[f'pred {title.strip(" OR()")}'] = df['pred']


    return predictions_df, metrics

def test_with_GT(logit_clf, X_df):
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


def get_nonparametric_preds(model, X, thresh=0.5):
    """Return probability of occupancy, based on the nonparametric model
    """
    # print('np preds: len X', len(X))
    X_np = X[['weekend']]
    X_np.insert(loc=1, column='time', value=X_np.index.time)
    df = X_np.merge(model, how='left', on=['weekend', 'time'])
    df.index = X_np.index
    df.drop(columns=['weekend', 'time'], inplace=True)
    df.columns = ['Probability']
    df['Predictions'] = df['Probability'].apply(lambda x: 1 if (x >= thresh) else 0)

    # print('len preds df', len(df))
    return df
    


def test_with_predictions(logit_clf, X, hr_lag=8, min_inc=5):
    """Run data through classifier and push predictions forward as lag values

    This is used instead of get_predictions_wGT.
    Returns: probabilities (between 0,1) and predictions (0/1) as a df

    """

    ts = int(60/min_inc)
    lag_max = hr_lag*ts

    X_start = X.iloc[:lag_max]
    lag_cols=[c for c in X.columns if c.startswith('lag')]
    exog_vars = X.drop(columns=lag_cols).iloc[lag_max:]
    preds_X = pd.concat([X_start, exog_vars])
    preds_X.index = pd.to_datetime(preds_X.index)

    ys = []
    k=0
    for idx, _ in preds_X.iterrows():
        k+=1
        df_row = preds_X.loc[idx]
        curr_row = df_row.to_numpy().reshape(1,-1)

        y_hat = logit_clf.predict(curr_row)
        y_proba = logit_clf.predict_proba(curr_row)[:,1]
        idx_loc = preds_X.index.get_loc(idx)
        
        ys.append((idx, y_proba[0], y_hat[0]))   

        if idx_loc < 11:
            continue

        to_avg = [y[1] for y in ys[-12:]]
        df_roll = np.mean(to_avg)

        for j in range(0, hr_lag):
            lag_col_name = f'lag{j+1}_occupied'
            ind_to_set = idx_loc + j*ts + 1
            try:
                preds_X.at[preds_X.iloc[ind_to_set].name, lag_col_name] = df_roll
            except:
                continue

    y_hats = pd.DataFrame(ys).set_index(0)
    y_hats.index.name = 'timestamp'
    y_hats.columns = ['Probability', 'Predictions']

    # drop_day = sorted(set(y_hats.index.date))[0]
    # y_hats = y_hats[y_hats.index.date != drop_day]

    return y_hats
