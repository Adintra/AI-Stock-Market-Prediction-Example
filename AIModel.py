import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Plotting datetime with matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class AIDataLoader:

    def __init__(self, dataset):
        # List of datasets available
        self.datasets = ["Tesla"]
        self.dataset = dataset
        # Default dataset for unknown dataset entered
        if dataset not in self.datasets:
            self.dataset = "Tesla"
        self.datapath = "Data/" + self.dataset + ".csv"

        # Pandas dataset
        self.df = pd.read_csv(self.datapath, index_col=0, parse_dates=True)
        self.df_initial_length = len(self.df)
        self.df.dropna(inplace=True)  # Drop missing values
        self.df_dropped_length = len(self.df)
        self.data = pd.DataFrame(self.df[' Close/Last'])  # Column names have extra space at start.
        self.data[self.data.columns[0]] = self.data[self.data.columns[0]].apply(lambda x: x.replace(' $', ''))

        # Prepare data for train/test/eval
        self.data_values = self.data[' Close/Last'].values.astype(float)
        self.test_size = 12
        # Train and test sets
        self.train_set = self.data_values[:-self.test_size]
        self.test_set = self.data_values[-self.test_size:]

        # Normalize data
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.train_norm = self.scaler.fit_transform(self.train_set.reshape(-1, 1))
        self.train_norm = torch.FloatTensor(self.train_norm).view(-1)
        self.train_data = self.input_data(window_size=12)

    def input_data(self, window_size):
        out = []
        length = len(self.train_norm)
        for i in range(length-window_size):
            window = self.train_norm[i:i+window_size]
            label = self.train_norm[i+window_size:i+window_size+1]
            out.append((window, label))
        return out


if __name__ == "__main__":
    tesla_dataloader = AIDataLoader("Tesla")
    print(tesla_dataloader.df.head())
    print(tesla_dataloader.df_initial_length)
    print(tesla_dataloader.df_dropped_length)
    print(tesla_dataloader.data.head())
    print(tesla_dataloader.test_set)
    print(tesla_dataloader.train_data[0])
