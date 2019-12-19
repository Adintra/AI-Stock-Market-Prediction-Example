import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Plotting datetime with matplotlib
from pandas.plotting import register_matplotlib_converters

import time

register_matplotlib_converters()

DATA_SETS = ["Tesla"]


class AIDataLoader:

    def __init__(self, data_set):
        # List of data sets available
        self.data_set = data_set
        # Default data set for unknown data set entered
        if data_set not in DATA_SETS:
            self.data_set = "Tesla"
        self.data_path = "Data/" + self.data_set + ".csv"

        # Pandas data set
        self.df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
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
        for i in range(length - window_size):
            window = self.train_norm[i:i + window_size]
            label = self.train_norm[i + window_size:i + window_size + 1]
            out.append((window, label))
        return out

    class LSTMnetwork(nn.Module):

        def __init__(self, input_size=1, hidden_size=100, output_size=1, data_set="Tesla"):
            super().__init__()
            self.hidden_size = hidden_size

            # LSTM layer
            self.lstm = nn.LSTM(input_size, hidden_size)
            # Fully connected layer
            self.linear = nn.Linear(hidden_size, output_size)
            # Initialize feedback of lstm neurons
            self.hidden = (torch.zeros(1, 1, self.hidden_size),
                           torch.zeros(1, 1, self.hidden_size))

            # Initialize data sets
            self.data_set = data_set
            if data_set not in DATA_SETS:
                self.data_set = "Tesla"
            self.data_loader = AIDataLoader(data_set)

            # Loss and optimization
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        def forward(self, seq):
            lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
            pred = self.linear(lstm_out.view(len(seq), -1))
            return pred[-1]  # Only need last value

        def train_network_train_data(self, epochs):
            start_time = time.time()
            for epoch in range(epochs):

                # Seq and label from training data
                for seq, y_train in self.data_loader.train_data:
                    # Parameter reset
                    self.optimizer.zero_grad()
                    self.hidden = (torch.zeros(1, 1, self.hidden_size),
                                   torch.zeros(1, 1, self.hidden_size))
                    y_prediction = self(seq)
                    loss = self.criterion(y_prediction, y_train)
                    loss.backward()
                    self.optimizer.step()

                # training result
                print(f'Epoch: {epoch + 1:2} Loss: {loss.item():10.8f}')

            # Duration
            print(f'\nDuration: {time.time() - start_time:.0f} seconds')


if __name__ == "__main__":
    tesla_dataloader = AIDataLoader("Tesla")
    print(tesla_dataloader.df.head())
    print(tesla_dataloader.df_initial_length)
    print(tesla_dataloader.df_dropped_length)
    print(tesla_dataloader.data.head())
    print(tesla_dataloader.test_set)
    print(tesla_dataloader.train_data[0])
