"""
etl.py
Author: Maggie Jacoby
Last update: 2021-02-22

Extract-Transform-Load class for logistic regression models 
Uses DataBasics as parent class

This script can be run alone, to create the csv train/test sets,
or it can be called in train.py, test.py, or explore.py to load and/or create the sets.

Outputs: train/test split dataframes (self.train and self.test)
"""

import os
import sys
import csv
import argparse
import logging
import pandas as pd
from glob import glob
from datetime import datetime, date

from data_basics import ModelBasics


class ETL(ModelBasics):
    """All functions to extract, transform, and load the train and test data sets.

    Upon initialization, this class gets the full names for all storage locations.     
    If a particular data type is requested (train or test),
    it either loads the data from existing csvs, 
    or (if none exist) loads the raw data, extracts the relevant information, and writes csvs.

    This class is used in train.py, test.py, and explore.py
    """

    def __init__(self, home, fill_type='zeros', data_type='train and test', log_flag=True):
        
        self.home = home
        # self.hub = hub
        self.fill_type = fill_type
        self.get_directories()
        self.configs = None
        self.days = []
        self.train, self.test = None, None
        self.log_flag = log_flag
        self.format_logs(log_type='ETL', home=self.home)
        
        self.load_data(data_type=data_type)


    def load_data(self, data_type):
        """Checks if the data type specified exists, decides how to load the data.

        If previous csvs exist for train/test, loads the one(s) requested,
        otherwise creates new ones and writes csvs.
        Sets values of self.train and/or self.test by reading in or create new.
        Can load train and/or test, but if creating, it creates both.

        Returns: Nothing
        """
        dt1 = data_type.split(' ')[0]
        check_name = os.path.join(self.data_dir, f'{dt1}_{self.home}_{self.fill_type}.csv')
        data_exists = os.path.exists(check_name)

        assert dt1 == 'train' or dt1 == 'test', 'Unrecognized data type'

        if data_exists:

            if data_type == 'train':
                self.train = self.read_csvs(data_type)
            elif data_type == 'test':
                self.test = self.read_csvs(data_type)
            else:
                self.train = self.read_csvs('train')
                self.test = self.read_csvs('test')

        else:
            print(f'\t>>> Data fill type {self.fill_type} for {self.home} does not exist. Creating new files...')
            df = self.create_new_datafiles()
            self.train, self.test = self.get_train_test(df)
            self.write_data()


    def read_csvs(self, data_type):
        """Reads in previously created train or test data.

        Returns: requested data file as pandas df
        """
        data_files = glob(os.path.join(self.data_dir, f'{data_type}_{self.home}_{self.fill_type}.csv'))
        if len(data_files) == 0:
            print(f'\t!!! No {data_type} files for {self.home}. Exiting program.')
            sys.exit()

        data_path = data_files[0]

        logging.info(f'Loading data file {os.path.basename(data_path)}')             
        print(f'\t>>> Loading data file {os.path.basename(data_path)}...')

        data_type_df = pd.read_csv(data_path, index_col='timestamp')
        return data_type_df


    def create_new_datafiles(self):
        """Sets up the conditions to read in raw inferences.

        This is only called if no test/train csv data exists yet.
        It reads configuration files and creates the list of days to use.
        Creates new train/test data through self.read_infs.

        Returns: pandas df (with all days)
        """ 
        config_file_list = glob(os.path.join(self.config_dir, f'H1_etl_*.yaml'))
        self.configs = self.read_config(config_files=config_file_list)
        self.days = self.get_days(self.configs['start_end'])

        data_path = os.path.join(self.raw_data, f'{self.home}_prob.csv')
        # data_path = os.path.join(self.raw_data, f'{self.home}_RS4_prob.csv')
        df = self.read_infs(data_path=data_path)
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


    def read_infs(self, data_path, resample_rate='5min', thresh=0.5):

        """ Reads in raw inferences from hub level CSV.

        This is called from self.crete_new_datafiles when no train/test data exist.
        This function fills or drops nans, resamples data, and calls create_lag function.

        Returns: pandas df
        """
        logging.info(f'Reading inferences from {os.path.basename(data_path)}')

        df = pd.read_csv(data_path, index_col="timestamp")
        df.index = pd.to_datetime(df.index)
        df = df.resample(rule=resample_rate).mean()
        df['occupied'] = df['occupied'].apply(lambda x: 1 if (x >= thresh) else 0)

        df = self.fill_df(df=df, fill_type=self.fill_type)
        df = self.create_lags(df)
        return df


    def create_lags(self, df, lag_hours=8, min_inc=5):
        """Creates lagged occupancy variable

        Takes in a df and makes lags up to (and including) lag_hours.
        The df is in 5 minute increments (by default), so lag is 12*hour.
        
        Returns: lagged df
        """
        ts = int(60/min_inc)
        occ_series = df['occupied']
        logging.info(f'Creating data with a lag of {lag_hours} hours')

        for i in range(1, lag_hours+1):
            lag_name = f'lag{i}_occupied'
            df[lag_name] = occ_series.shift(periods=ts*i)
        return df


    def get_train_test(self, DF):
        """Splits data into train and test sets.

        Data is time series, so splits on day, not randomly.
        Also subsets based on the list of days specified by self.days.

        Returns: training set and testing set
        """
        df = DF.copy()
        df['date'] = pd.to_datetime(df.index) 
        df.insert(loc=0, column='day', value=df['date'].dt.date)
        df.drop(columns=['date'], inplace=True)

        train_size = int(len(self.days) * 0.7) - 1
        train_days = self.days[ :train_size]
        test_days = self.days[train_size: ]
        train_days = sorted([datetime.strptime(day_str, '%Y-%m-%d').date() for day_str in train_days])
        test_days = sorted([datetime.strptime(day_str, '%Y-%m-%d').date() for day_str in test_days])

        logging.info(f'Training: {len(train_days)} days from {train_days[0]} to {train_days[-1]}')
        logging.info(f'Testing: {len(test_days)} days from {test_days[0]} to {test_days[-1]}')
        
        train_df = df[df['day'].isin(train_days)]
        test_df = df[df['day'].isin(test_days)]
        return train_df, test_df


    def get_days(self, start_end):
        """Gets all days to use for the training or testing.

        If multiple lists of start/end exist, it joins them together. 
        Returns: a list of all days between start/end in config file.
        """
        all_days = []

        for st in start_end:
            start, end = st[0], st[1]
            pd_days = pd.date_range(start=start, end=end).tolist()
            days = [d.strftime('%Y-%m-%d') for d in pd_days]
            all_days.extend(days)

        logging.info(f'{len(all_days)} days from {len(start_end)} continuous period(s)')
        return sorted(all_days)


    def write_data(self):
        """Writes csvs with the newly created train/test data.

        returns: nothing
        """
        os.makedirs(self.data_dir, exist_ok=True)   
        train_fname = os.path.join(self.data_dir, f'train_{self.home}_{self.fill_type}.csv')
        test_fname = os.path.join(self.data_dir, f'test_{self.home}_{self.fill_type}.csv')
        logging.info(f'Writing data to {os.path.basename(train_fname)} and {os.path.basename(test_fname)}')

        self.train.to_csv(train_fname, index_label='timestamp')
        self.test.to_csv(test_fname, index_label='timestamp')


    def split_xy(self, df):
        """Split dataset to get predictors (X) and ground truth (y)

        Returns: X: pandas df, and y: pandas Series
        """ 
        y = df['occupied']
        X = df[df.columns.difference(['occupied'], sort=False)]
        X = X.drop(columns = ['day'])
        return X, y
        

    def combine_hubs(self):
        """Write function to take in raw inferences for all hubs specfied in config file.
        """
        pass


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract, transform, and load training/testing data')
    parser.add_argument('-home', '--home', default='H1', type=str, help='Home to get data for, eg H1')
    parser.add_argument('-data_type', '--data_type', default='train and test', type=str, help='Data type to load (if only one)')
    parser.add_argument('-fill_type', '--fill_type', default='zeros', type=str, help='How to treat missing values')
    # parser.add_argument('-hub', '--hub', default='RS4', type=str, help='Which hub to use')
    args = parser.parse_args()

    Data = ETL(
            home=args.home,
            data_type=args.data_type,
            fill_type=args.fill_type,
            # hub=args.hub,
            log_flag=False
            )