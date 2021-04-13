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


class ETL():
    """All functions to extract, transform, and load the train and test data sets.

    Upon initialization, this class gets the full names for all storage locations.     
    If a particular data type is requested (train or test),
    it either loads the data from existing csvs, 
    or (if none exist) loads the raw data, extracts the relevant information, and writes csvs.

    This class is used in train.py, test.py, and explore.py
    """

    def __init__(self, H_num, fill_type='zeros'):
        self.H_num = H_num
        self.fill_type = fill_type
        self.get_directories()


    def generate_dataset(self, hub=''):
        self.home_configs = self.read_config(config_type='etl')
        self.hubs_to_use = self.get_hubs(hub)
        self.days = self.get_days()

        df = self.get_data()
        self.train, self.test = self.get_train_test(df)


    def get_directories(self):
        """Sets names of all directories, relative to the current script location.

        Returns: nothing
        """
        parent_dir = os.path.dirname(os.getcwd())

        self.config_dir = os.path.join(parent_dir, 'configuration_files')
        self.log_save_dir = os.path.join(parent_dir, 'logs')
        self.data_dir = os.path.join(parent_dir, 'data')
        self.models_dir = os.path.join(parent_dir, 'models')
        self.raw_data = os.path.join(parent_dir, 'raw_data_files')
        self.results_csvs = os.path.join(parent_dir, 'results_csvs')


    def read_config(self, config_type, config_file=None):
        """Reads in the configuration file (*.yaml).
        
        Returns: configuration parameters
        """
        if not config_file:
            file_list = glob(os.path.join(self.config_dir, f'{self.H_num}_{config_type}_config.yaml'))

            if len(file_list) == 0:
                print(f'No {config_type} configuration file for {self.H_num}. Exiting program.')
                sys.exit()

            config_file_path = file_list[0]
            print(f'{len(file_list)} {config_type} configuration file(s) found for {self.H_num}')

        else:
            config_file_path = os.path.join(self.config_dir, f'{config_file}.yaml')

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
        all_days = []

        for st in self.home_configs['start_end']:
            start, end = st[0], st[1]
            pd_days = pd.date_range(start=start, end=end).tolist()
            days = [d.strftime('%Y-%m-%d') for d in pd_days]
            all_days.extend(days)

        return sorted(all_days)


    def get_data(self):
        """Reads raw inferences for all hubs, creates dfs (including lags) and combines hubs.

        Returns: pandas df (with all days)
        """ 

        print(f'Creating new datasets for {self.H_num}...')
        all_hub_dfs = []

        for hub in self.hubs_to_use:
            hub_path = os.path.join(self.raw_data, self.H_num, f'{self.H_num}{hub}_prob.csv')
            hub_df = self.read_infs(data_path=hub_path)
            all_hub_dfs.append(hub_df)

        df = all_hub_dfs[0] ###
        # df = self.combine_hubs(df_list=all_hub_dfs)
        df = self.create_HOD(df)
        df = self.create_lags(df)
        return df


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


    def get_train_test(self, DF):
        """Splits data into train and test sets.

        Data is time series, so splits on day, not randomly.
        Also subsets based on the list of days specified by self.days.

        Returns: training set and testing set
        """
        df = DF.copy()

        train_size = int(len(self.days) * 0.6)
        train_days = self.days[ :train_size]
        test_days = self.days[train_size: ]
        train_days = sorted([datetime.strptime(day_str, '%Y-%m-%d').date() for day_str in train_days])
        test_days = sorted([datetime.strptime(day_str, '%Y-%m-%d').date() for day_str in test_days])

        print('Train days:', len(train_days))
        print('Test days:', len(test_days))
        
        train_df = df[df['day'].isin(train_days)]
        test_df = df[df['day'].isin(test_days)]

        return train_df, test_df


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
        df['weekday'] = 0
        df.loc[df.DOW > 4, 'weekday'] = 1

        df['hr_sin'] = np.sin(df.hour*(2.*np.pi/24))
        df['hr_cos'] = np.cos(df.hour*(2.*np.pi/24))

        df.drop(columns=['date', 'DOW', 'hour'], inplace=True)
        return df


    def create_lags(self, df, lag_hours=8, min_inc=5):
        """Creates lagged occupancy variable

        Takes in a df and makes lags up to (and including) lag_hours.
        The df is in 5 minute increments (by default), so lag is 12*hour.
        
        Returns: lagged df
        """
        ts = int(60/min_inc)
        occ_series = df['occupied']

        # occ_rolling_avg = 

        for i in range(1, lag_hours+1):
            lag_name = f'lag{i}_occupied'
            df[lag_name] = occ_series.shift(periods=ts*i)
        return df


    def get_rolling_avg(self, df):
        """Write function that gets lag as a rolling average, not discrete.

        Returns: df with rolling averages
        """
        pass


    def combine_hubs(self, df_list):
        """Write function to combine multiple hub dfs into one. 
        Takes in list of dfs

        Returns: df with all hubs combined
        """        
        pass


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract, transform, and load training/testing data')
    parser.add_argument('-home', '--home', default='H1', type=str, help='Home to get data for, eg H1')
    parser.add_argument('-hub', '--hub', default='', type=str, help='which hub to use? (leave blank if using config file to specify)')
    parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')
    args = parser.parse_args()

    Data = ETL(
            H_num=args.home,
            # hub=args.hub,
            fill_type=args.fill_type
            )
    Data.generate_dataset(hub=args.hub)