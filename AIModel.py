import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    tesla_dataloader = AIDataLoader("Tesla")
    print(tesla_dataloader.df.head())
    print(tesla_dataloader.df_initial_length)
    print(tesla_dataloader.df_dropped_length)
    print(tesla_dataloader.data.head())
