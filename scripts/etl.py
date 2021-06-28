"""
etl.py
Author: Maggie Jacoby
Last update: 2021-04-12

Extract-Transform-Load class for logistic regression models 
Uses DataBasics as parent class

This script can be run alone, to create the csv train/test sets,
or it can be called in train.py, test.py, or explore.py to load and/or create the sets.

Outputs: train/test split dataframes (self.train and self.test)
"""

import os
import sys
import csv
import yaml
import argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, date

import prediction_functions as pred_fncs


class ETL():
    """All functions to extract, transform, and load the train and test data sets.

    Upon initialization, this class gets the full names for all storage locations.     
    If a particular data type is requested (train or test),
    it either loads the data from existing csvs, 
    or (if none exist) loads the raw data, extracts the relevant information, and writes csvs.

    This class is used in train.py, test.py, and explore.py
    """

    def __init__(self, H_num, fill_type='zeros', g=8):
        self.H_num = H_num
        self.fill_type = fill_type
        self.g = g
        self.get_directories()


    def generate_dataset(self, hub=''):
        self.home_configs = self.read_config(config_type='etl')
        self.hubs_to_use = self.get_hubs(hub)

        self.groups, self.all_days = self.get_days()
        self.df = self.get_data()
        self.daysets = self.get_groups(df=self.df, g=self.g)
        # print(len(self.daysets))
        # sys.exit() 
        # self.train, self.test = self.get_train_test(self.df)


    def get_directories(self):
        """Sets names of all directories, relative to the current script location.

        Returns: nothing
        """
        parent_dir = os.getcwd()

        self.config_dir = os.path.join(parent_dir, 'configuration_files')
        self.log_save_dir = os.path.join(parent_dir, 'logs')
        self.data_dir = os.path.join(parent_dir, 'data')
        self.raw_data = os.path.join(parent_dir, 'raw_data_files')
        self.models_dir = os.path.join(parent_dir, 'models', self.H_num)
        self.results_csvs = os.makedirs(os.path.join(parent_dir, 'Results', self.H_num), exist_ok=True)


    def read_config(self, config_type, config_file=None):
        """Reads in the configuration file (*.yml).
        
        Returns: configuration parameters
        """
        if not config_file:
            file_list = glob(os.path.join(self.config_dir, f'{self.H_num}_{config_type}_config.yml'))
            print(os.path.join(self.config_dir, f'{self.H_num}_{config_type}_config.yml'))

            if len(file_list) == 0:
                print(f'No {config_type} configuration file for {self.H_num}. Exiting program.')
                sys.exit()

            config_file_path = file_list[0]
            print(f'{len(file_list)} {config_type} configuration file(s) found for {self.H_num}')

        else:
            config_file_path = os.path.join(self.config_dir, f'{config_file}.yml')

        print(f'Using: {os.path.basename(config_file_path)}')

        with open(config_file_path) as f:
            config = yaml.safe_load(f)

        return config


    def get_hubs(self, hub):
        """Returns a list of the hubs to use.
        If a hub is initially specified, returns just that as a list,
        otherwise looks in configuration file and creates a list
        """

        if len(hub) > 0:
            hubs_to_use = [hub]

        else:
            color = self.home_configs['H_system'][0].upper()
            hubs_to_use = []
            for num in self.home_configs['hubs']:
                hubs_to_use.append(f'{color}S{num}')

        print(f'Using hubs: {hubs_to_use}')

        return hubs_to_use


    def get_days(self):
        """Gets all days to use for the training or testing.

        If multiple lists of start/end exist, it joins them together. 
        Returns: a list of all days between start/end in config file.
        """
        groups = []

        for st in self.home_configs['start_end']:
            start, end = st[0], st[1]
            pd_days = pd.date_range(start=start, end=end).tolist()
            days = [d.strftime('%Y-%m-%d') for d in pd_days]
            groups.append(days)

        all_days = []
        for d in groups:
            all_days += d

        return groups, all_days


    def get_groups(self, df, g=8):
        daysets = []
        for grp in self.groups:
            for i in range(0,int(np.round(len(grp)/g))):
                nw_gp = grp[i*g:(i+1)*g]
                days = sorted([datetime.strptime(day_str, '%Y-%m-%d').date() for day_str in nw_gp])
                gp_df = df[df['day'].isin(days)]
                daysets.append(gp_df)
        return daysets
            

    def get_data(self):
        """Reads raw inferences for all hubs, creates dfs (including lags) and combines hubs.

        Returns: pandas df (with all days)
        """ 

        print(f'Creating new datasets for {self.H_num}...')
        all_hub_dfs = []

        for hub in self.hubs_to_use:
            hub_path = os.path.join(self.raw_data, self.H_num, f'{self.H_num}{hub}.csv')
            hub_df = self.read_infs(data_path=hub_path)
            all_hub_dfs.append(hub_df)

        df = pd.concat(all_hub_dfs).groupby(level=0).max()

        df = self.create_HOD(df)
        df = self.create_rolling_lags(df)
        print(len(df['day'].unique()))
        days = sorted([datetime.strptime(day_str, '%Y-%m-%d').date() for day_str in self.all_days])

        df_dates = df[df['day'].isin(days)]
        print(len(df_dates['day'].unique()))
        # sys.exit()
        # df["Date"].map(pd.Timestamp.date).unique()
        # df.to_csv('~/Desktop/test_6-26/test_df.csv')
        return df_dates


    def read_infs(self, data_path, resample_rate='5min', thresh=0.5):

        """ Reads in raw inferences from hub level CSV.
        This function fills or drops nans, resamples data, and calls create_lag function.

        Returns: pandas df
        """
        df = pd.read_csv(data_path, index_col='timestamp')
        df.index = pd.to_datetime(df.index)
        df = df.resample(rule=resample_rate).mean()
        df['occupied'] = df['occupied'].apply(lambda x: 1 if (x >= thresh) else 0)

        df = self.fill_df(df=df, fill_type=self.fill_type)
        return df


    def fill_df(self, df, fill_type):

        if fill_type == 'zeros':
            df = df.fillna(0)
        elif fill_type == 'ones':
            df = df.fillna(1)
        elif fill_type == 'ffill':
            df = df.fillna(method='ffill')
            df = df.fillna(0)
        else:
            print(f'\t!!! Unrecognized fill type {fill_type}. Exiting program.')
            sys.exit()
        return df


    # def get_train_test(self, DF):
    #     """Splits data into train and test sets.

    #     Data is time series, so splits on day, not randomly.
    #     Also subsets based on the list of days specified by self.days.

    #     Returns: training set and testing set
    #     """
    #     df = DF.copy()

    #     train_size = int(len(self.days) * 0.6)
    #     train_days = self.days[ :train_size]
    #     test_days = self.days[train_size: ]
    #     train_days = sorted([datetime.strptime(day_str, '%Y-%m-%d').date() for day_str in train_days])
    #     test_days = sorted([datetime.strptime(day_str, '%Y-%m-%d').date() for day_str in test_days])

    #     # print('Train days:', len(train_days))
    #     # print('Test days:', len(test_days))
        
    #     # print('train days: ')
    #     # trx=0
    #     # for i, day in enumerate(train_days,1):
    #     #     print(day)
    #     #     if i%3 == 0:
    #     #         print('*****')
    #     #         trx += 1
    #     # print('*****', trx, 'groups')  

    #     # print('Test days:')
    #     # tsx = 0
    #     # for i, day in enumerate(test_days,1):
    #     #     print(day)
    #     #     if i%3 == 0:
    #     #         tsx += 1
    #     #         print('*****')
    #     # print('*****', tsx, 'groups')  

    #     train_df = df[df['day'].isin(train_days)]
    #     test_df = df[df['day'].isin(test_days)]

    #     return train_df, test_df


    def split_xy(self, df):
        """Split dataset to get predictors (X) and occupancy (y)

        Returns: X: pandas df, and y: pandas Series
        """ 
        y = df['occupied']
        X = df[df.columns.difference(['occupied'], sort=False)]
        X = X.drop(columns = ['day'])
        return X, y


    def create_HOD(self, df):
        """Creates variables for hour of day (cyclic) and weekday/weekend (binary).

        Returns: df of same length, with 4 new variables (day, hr_sin, hr_cos, weekday)
        """

        df['date'] = pd.to_datetime(df.index) 
        df.insert(loc=0, column='day', value=df['date'].dt.date)
        df.insert(loc=1, column='hour', value=df['date'].dt.hour+df['date'].dt.minute/60)
        df.insert(loc=2, column='DOW', value=df['date'].dt.weekday)

        df['weekend'] = 0
        df.loc[df.DOW > 4, 'weekend'] = 1

        df['hr_sin'] = np.sin(df.hour*(2.*np.pi/24))
        df['hr_cos'] = np.cos(df.hour*(2.*np.pi/24))

        df.drop(columns=['date', 'DOW', 'hour'], inplace=True)
        return df


    # def create_static_lags(self, df, lag_hours=8, min_inc=5):
    #     """Creates lagged occupancy variable

    #     Takes in a df and makes lags up to (and including) lag_hours.
    #     The df is in 5 minute increments (by default), so lag is 12*hour.
        
    #     Returns: lagged df
    #     """
    #     ts = int(60/min_inc)
    #     occ_series = df['occupied']

    #     for i in range(1, lag_hours+1):
    #         lag_name = f'lag{i}_occupied'
    #         df[lag_name] = occ_series.shift(periods=ts*i)
    #     return df


    def create_rolling_lags(self, df, lag_hours=8, min_inc=5):

        ts = int(60/min_inc)
        df_roll = df['occupied'].rolling(window=ts).mean()

        for i in range(0, lag_hours):
            lag_name = f'lag{i+1}_occupied'
            df[lag_name] = df_roll.shift(periods=ts*i+1)
            df[lag_name] = df[lag_name].round()
        return df


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract, transform, and load training/testing data')
    parser.add_argument('-home', '--home', default='H1', type=str, help='Home to get data for, eg H1')
    parser.add_argument('-hub', '--hub', default='', type=str, help='which hub to use? (leave blank if using config file to specify)')
    parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')
    args = parser.parse_args()

    Data = ETL(
            H_num=args.home,
            fill_type=args.fill_type
            )
    Data.generate_dataset(hub=args.hub)


