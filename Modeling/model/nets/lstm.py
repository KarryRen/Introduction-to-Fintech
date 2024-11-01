# -*- coding: utf-8 -*-
# @Time    : 2024/4/12 18:18
# @Author  : Karry Ren

""" The Comparison Methods 2: LSTM.

Ref. https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_lstm.py#L286

"""

import logging
import torch
from torch import nn


class LSTM_Net(nn.Module):
    """ The 2 Layer LSTM. hidden_size=64. """

    def __init__(
            self, input_size: int, hidden_size: int = 64, out_size: int = 3, num_layers: int = 2,
            dropout: float = 0.0, device: torch.device = torch.device("cpu")
    ):
        """ The init function of LSTM_Net Net.

        :param input_size: input size for each time step
        :param hidden_size: hidden size of lstm
        :param num_layers: the num of lstm layers
        :param dropout: the dropout ratio
        :param device: the computing device

        """

        super(LSTM_Net, self).__init__()
        self.device = device

        # ---- Log the info of Multi-Grained LSTM ---- #
        logging.info(
            f"|||| Using LSTM Now ! input_size={input_size}, "
            f"hidden_size={hidden_size}, num_layers={num_layers}, dropout_ratio={dropout}||||"
        )

        # ---- Part 1. The LSTM module ---- #
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout
        ).to(device=device)

        # ---- Part 2. The output fully connect layer ---- #
        self.fc = nn.Linear(hidden_size, out_size).to(device=device)

    def forward(self, feature_input: torch.Tensor) -> torch.Tensor:
        """ The forward function of MLP Net.

        :param feature_input: input feature, shape=(bs, time_steps, input_size)

        returns: output: the prediction, which is a tensor of shape (bs, 1)

        """

        # ---- Forward computing ---- #
        hidden_x, _ = self.lstm(feature_input)  # shape=(bs, time_steps, hidden_size)
        last_step_hidden_x = hidden_x[:, -1, :]  # shape=(bs, hidden_size)
        output = self.fc(last_step_hidden_x)  # shape=(bs, 1)

        # ---- Return the result ---- #
        return output


if __name__ == "__main__":  # A demo using LSTM_Net
    bath_size, time_steps, feature_dim = 64, 1, 167
    hidden_size = 64
    feature = torch.ones((bath_size, time_steps, feature_dim))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_Net(input_size=time_steps * feature_dim, hidden_size=64, device=dev)
    out = model(feature)
    print(out.shape)
