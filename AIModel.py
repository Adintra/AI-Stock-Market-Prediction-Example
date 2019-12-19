import torch
import torch.nn as nn

# Plotting datetime with matplotlib
from pandas.plotting import register_matplotlib_converters
import time
from AI_Data_Loader import AIDataLoader

register_matplotlib_converters()
DATA_SETS = ["Tesla"]


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
