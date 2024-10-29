# -*- coding: utf-8 -*-
# @Time    : 2024/10/20 15:30
# @Author  : Karry Ren

""" The LSTM Model. """

import logging
import torch
from torch import nn


class LSTM_Net(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, out_size: int = 3, device: torch.device = torch.device("cpu")):
        """ The init function of MLP Net.

        :param input_size: input size for each time step
        :param hidden_size: hidden size
        :param device: the computing device

        """

        super(LSTM_Net, self).__init__()
        self.device = device

        # ---- Log the info of MLP ---- #
        logging.info(f"|||| Using LSTM_Net Now ! input_size={input_size}, hidden_size={hidden_size}")

        # ---- Build up the model ---- #
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True).to(device=device)
        self.fc = nn.Linear(in_features=hidden_size, out_features=out_size).to(device=device)

    def forward(self, feature_input: torch.Tensor) -> torch.Tensor:
        """ The forward function of MLP Net.

        :param feature_input: input feature, shape=(bs, time_steps, input_size)

        returns: output: the prediction, which is a tensor of shape (bs, 1)

        """

        # ---- Forward computing ---- #
        hidden_x, _ = self.lstm(feature_input)  # shape=(bs, T, hidden_size)
        last_hidden_x = hidden_x[:, -1, :]  # shape=(bs, hidden_size)
        output = self.fc(last_hidden_x)  # shape=(bs, 1)

        # ---- Return the result ---- #
        return output


if __name__ == "__main__":  # A demo using MLP_Net
    bath_size, time_steps, feature_dim = 64, 2, 26
    hidden_size = 64
    feature = torch.ones((bath_size, time_steps, feature_dim))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_Net(input_size=feature_dim, hidden_size=64, device=dev)
    out = model(feature)
    print(out.shape)
