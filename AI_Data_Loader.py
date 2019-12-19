import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


class AIDataLoader:

    def __init__(self, data_set):
        # List of data sets available
        self.data_set = data_set
        # Default data set for unknown data set entered
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
