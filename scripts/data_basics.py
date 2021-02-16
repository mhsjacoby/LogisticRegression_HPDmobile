"""
data_basics.py
Author: Maggie Jacoby
February, 2021

Parent class for ETL, TrainModel, and TestModel
contains basic functions for logging, getting configuration files, getting storage directories, etc

"""

import os
import sys
import csv
# import json
import yaml
import argparse
from glob import glob
import pandas as pd
from datetime import datetime, date
import logging


class DataBasics():

    def get_directories(self):
        parent_dir = os.path.dirname(os.getcwd())

        self.config_dir = os.path.join(parent_dir, 'configuration_files')
        self.log_save_dir = os.path.join(parent_dir, 'logs')
        self.data_dir = os.path.join(parent_dir, 'data')
        self.models_dir = os.path.join(parent_dir, 'models')
        self.raw_data = os.path.join(parent_dir, 'raw_data_files')


    def read_config(self, config_files):
        """
        reads in the configuration file
        returns: configuration parameters
        """

        if len(config_files) == 0:
            print(f'No ETL configuration file for {self.home}. Exiting program.')
            sys.exit()

        config_file_path = config_files[0]
        logging.info(f'{len(config_files)} ETL configuration file(s).')
        logging.info(f'Using: {os.path.basename(config_file_path)}')

        with open(config_file_path) as f:
            # config = json.load(f)
            config = yaml.safe_load(f)

        return config


    def format_logs(self, log_type, home):

        os.makedirs(self.log_save_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(self.log_save_dir, f'{log_type}_{home}.log'),
            level=logging.INFO,
            format='%(message)s',
            datefmt='%Y-%m-%d'
            )

        logging.info(f'\n\t\t##### NEW RUN #####\n{date.today()}')


    def get_days(self, start_end):
        """
        Returns: a list of all days between start/end in config file
        """

        all_days = []

        for st in start_end:
            start, end = st[0], st[1]
            pd_days = pd.date_range(start=start, end=end).tolist()
            days = [d.strftime('%Y-%m-%d') for d in pd_days]
            all_days.extend(days)

        logging.info(f'{len(all_days)} days, {len(start_end)} continuous period(s) \n{sorted(all_days)}')
        return sorted(all_days)
