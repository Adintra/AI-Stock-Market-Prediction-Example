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
        self.df_length = len(self.df)


if __name__ == "__main__":
    tesla_dataloader = AIDataLoader("Tesla")
    print(tesla_dataloader.df.head())
