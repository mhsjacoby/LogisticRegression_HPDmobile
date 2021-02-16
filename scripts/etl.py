"""
etl.py
Author: Maggie Jacoby
Base code provided by Jasmine Garland
February, 2021

Extract-Transform-Load class for logistic regression models 
Uses DataBasics as parent class

This script can be run alone, to create the csv train/test sets,
or it can be called in train.py or test.py to load and/or create the sets.

Outputs: train/test split dataframes (self.train and self.test)

TODO: 
    - Write function to combine multiple hubs
    #//- Modify load_data/read_in_csvs to read in test and train data
    #// - Write logging function
    #// - Write function to read in existing train/test files
    #// - Write function to write train/test data to file
    #// - Make data_path take in multiple files
"""

import os
import sys
import csv
import json
import argparse
from glob import glob
import pandas as pd
from datetime import datetime, date
import logging

from data_basics import DataBasics


class ETL(DataBasics):
    """
    Upon initialization, this class gets the full names for all storage locations, 
    then checks to see if the requested data type exists (train or test).
    
    If a particular data type is requested (train or test),
    it either loads the data from existing csvs, 
    or loads the raw data, extracts the relevant information, and writes csvs.
    """

    def __init__(self, home, data_type='train and test'):

        self.home = home
        self.get_directories()

        self.configs = None
        self.days = []
        self.train, self.test = None, None

        self.load_data(data_type=data_type)


    def load_data(self, data_type):
        """
        Checks if the data type specified exists.
        Sets values of self.train and/or self.test
        If previous data exists (as csv), it loads them, else creates new and writes csvs.

        returns: Nothing
        """

        dt1 = data_type.split(" ")[0]

        check_name = os.path.join(self.data_dir, f'{dt1}_{self.home}.csv')
        data_exists = os.path.exists(check_name)

        if data_exists:

            if data_type == 'train':
                self.train = self.read_csvs(data_type)

            elif data_type == 'test':
                self.test = self.read_csvs(data_type)

            elif dt1 == 'train' or dt1 == 'test':
                self.train = self.read_csvs('train')
                self.test = self.read_csvs('test')

            else:
                print(f'Unknown datatype {data_type}. Exiting program.')
                sys.exit()

        else:

            if dt1 == 'train' or dt1 == 'test':
                df = self.create_new_datafiles()
                self.train, self.test = self.get_train_test(df)
                self.write_data()

            else:
                print(f'Unknown datatype {data_type}. Exiting program.')
                sys.exit()


    def read_csvs(self, data_type):
        """
        Reads in previously created train or test data

        returns: requested data file as pandas df
        """

        data_files = glob(os.path.join(self.data_dir, f'{data_type}_{self.home}.csv'))

        if len(data_files) == 0:
            print(f'No {data_type} files for {self.home}. Exiting program.')
            sys.exit()

        data_path = data_files[0]

        if len(data_files) > 1:
            print(f'{len(data_files)} {data_type} files for {self.home}.\nUsing: {os.path.basename(data_path)}.')

        print(f'Loading {data_type} data...')
        data_type_df = pd.read_csv(data_path, index_col='timestamp')

        return data_type_df


    def create_new_datafiles(self):
        """
        If no data files exist, reads configuration files,
        creates new train/test data

        returns: full df
        """
        
        config_files = glob(os.path.join(self.config_dir, f'{self.home}_etl_*.yaml'))
        
        self.format_logs(log_type='ETL', home=self.home)
        self.configs = self.read_config(config_files=config_files)

        data_path = os.path.join(self.raw_data, f'{home}_RS4_prob.csv')

        df = self.read_infs(data_path=data_path)
        self.days = self.get_days(self.configs['start_end'])

        return df


    def read_infs(self, data_path, fill_nan=True, resample_rate='5min', thresh=0.5):
        """
        if no test/train data exists yet, reads in raw inferences
        fills or drops nans and resamples
        calls create lag function

        returns: df
        """

        logging.info(f'Reading inferences from {data_path}')

        df = pd.read_csv(data_path, index_col="timestamp")
        df.index = pd.to_datetime(df.index)
        df = df.resample(rule=resample_rate).mean()
        df['occupied'] = df['occupied'].apply(lambda x: 1 if (x >= thresh) else 0)
    
        if fill_nan:
            df = df.fillna(0)
        else:
            df = df.dropna()

        df = self.create_lags(df)

        return df


    def create_lags(self, df, lag_hours=8):
        """
        creates lagged occupancy variable for every hour
        up to (and including) lag_hours
        df is in 5 minute increments, so lag is 12*hour
        
        return: lagged df
        """

        occ_series = df['occupied']
        logging.info(f'Creating data with a lag of {lag_hours} hours.')

        for i in range(1, lag_hours+1):
            lag_name = f'lag{i}_occupied'
            df[lag_name] = occ_series.shift(periods=12*i)

        return df


    def get_train_test(self, DF):
        """
        splits data into train and test sets
        data is time series, so splits on day, not randomly

        returns: training set and testing set
        """

        df = DF.copy()
        df['date'] = pd.to_datetime(df.index) 
        df.insert(loc=0, column='day', value=df['date'].dt.date)
        df.drop(columns=['date'], inplace=True)

        train_size = int(len(self.days) * 0.7) - 1
        train_days = self.days[ :train_size]
        test_days = self.days[train_size: ]
        train_days = [datetime.strptime(day_str, '%Y-%m-%d').date() for day_str in train_days] 
        test_days = [datetime.strptime(day_str, '%Y-%m-%d').date() for day_str in test_days] 

        train_df = df[df['day'].isin(train_days)]
        test_df = df[df['day'].isin(test_days)]

        logging.info(f'Training: {len(train_days)} days with {len(train_df)} datapoints.')
        logging.info(f'Testing: {len(test_days)} days with {len(test_df)} datapoints.')

        return train_df, test_df


    def write_data(self):
        """
        Writes csvs with the newly created train/test data

        returns: nothing
        """

        os.makedirs(self.data_dir, exist_ok=True)
    
        train_fname = os.path.join(self.data_dir, f'train_{self.home}.csv')
        test_fname = os.path.join(self.data_dir, f'test_{self.home}.csv')

        self.train.to_csv(train_fname, index_label='timestamp')
        self.test.to_csv(test_fname, index_label='timestamp')
        

    def combine_hubs(self):
        pass


    def main(self):
        pass



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract, transform, and load training/testing data')
    parser.add_argument('-home', '--home', default='H1', type=str, help='Home to get data for, eg H1')
    parser.add_argument('-data_type', '--data_type', default='train and test', type=str, help='Data type to load (if only one)')
    args = parser.parse_args()
    
    home = args.home
    data_type = args.data_type

    Data = ETL(home, data_type=data_type)
    print(Data.configs)
    print(Data.test.columns)


    
