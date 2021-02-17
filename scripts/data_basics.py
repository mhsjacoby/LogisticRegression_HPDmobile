"""
data_basics.py
Author: Maggie Jacoby
Last update: 2021-02-17
"""

import os
import sys
import csv
import yaml
import logging
import argparse
import pandas as pd
from glob import glob
from datetime import datetime, date


class DataBasics():
    """Parent class for ETL, TrainModel, and TestModel.
    
    Contains basic functions for logging, getting configuration files, 
    getting storage directory locations, and getting list of days.
    """ 

    def get_directories(self):
        """Gets names of all directories, relative to the current script.

        Returns: nothing
        """
        parent_dir = os.path.dirname(os.getcwd())

        self.config_dir = os.path.join(parent_dir, 'configuration_files')
        self.log_save_dir = os.path.join(parent_dir, 'logs')
        self.data_dir = os.path.join(parent_dir, 'data')
        self.models_dir = os.path.join(parent_dir, 'models')
        self.raw_data = os.path.join(parent_dir, 'raw_data_files')

    def read_config(self, config_files, config_type='ETL'):
        """Reads in the configuration file (*.yaml).
        
        Returns: configuration parameters
        """




        if len(config_files) == 0:
            print(f'No {config_type} configuration file for {self.home}. Exiting program.')
            logging.info('No configuration file.')
            sys.exit()

        config_file_path = config_files[0]
        logging.info(f'{len(config_files)} {config_type} configuration file(s).\
                    \nUsing: {os.path.basename(config_file_path)}')

        with open(config_file_path) as f:
            config = yaml.safe_load(f)

        return config

    def format_logs(self, log_type, home):
        """Creates log object.

        Returns: nothing
        """
        os.makedirs(self.log_save_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(self.log_save_dir, f'{log_type}_{home}.log'),
            level=logging.INFO,
            format='%(message)s',
            datefmt='%Y-%m-%d',
            )
        logging.info(f'\n\t\t##### NEW RUN #####\n{date.today()}')

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

        logging.info(f'{len(all_days)} days, {len(start_end)} continuous period(s) \n{sorted(all_days)}')
        return sorted(all_days)


