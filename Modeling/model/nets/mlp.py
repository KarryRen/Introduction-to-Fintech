# -*- coding: utf-8 -*-
# @Time    : 2024/10/20 11:33
# @Author  : Karry Ren

""" The benchmark MLP Model. """

import logging
import torch
from torch import nn


class MLP_Net(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, out_size: int = 3, device: torch.device = torch.device("cpu")):
        """ The init function of MLP Net.

        :param input_size: input size for each time step
        :param hidden_size: hidden size
        :param device: the computing device

        """

        super(MLP_Net, self).__init__()
        self.device = device

        # ---- Log the info of MLP ---- #
        logging.info(f"|||| Using MLP Now ! input_size={input_size}, hidden_size={hidden_size}")

        # ---- Build up the model ---- #
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size * 2, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False),
            nn.ReLU()
        ).to(device=device)
        self.fc = nn.Linear(in_features=hidden_size, out_features=out_size).to(device=device)

    def forward(self, feature_input: torch.Tensor) -> torch.Tensor:
        """ The forward function of MLP Net.

        :param feature_input: input feature, shape=(bs, time_steps, input_size)

        returns: output: the prediction, which is a tensor of shape (bs, 1)

        """

        # ---- Reshape ---- #
        bs = feature_input.shape[0]
        x = feature_input.reshape(bs, -1)  # shape=(bs, time_steps * input_size)

        # ---- Forward computing ---- #
        x = self.mlp(x)  # shape=(bs, hidden_size)
        output = self.fc(x)  # shape=(bs, 1)

        # ---- Return the result ---- #
        return output


class Big_MLP_Net(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, out_size: int = 3, device: torch.device = torch.device("cpu")):
        """ The init function of MLP Net.

        :param input_size: input size for each time step
        :param hidden_size: hidden size
        :param device: the computing device

        """

        super(Big_MLP_Net, self).__init__()
        self.device = device

        # ---- Log the info of MLP ---- #
        logging.info(f"|||| Using MLP Now ! input_size={input_size}, hidden_size={hidden_size}")

        # ---- Build up the model ---- #
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size * 2, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size * 2, out_features=hidden_size * 2, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False),
            nn.ReLU()
        ).to(device=device)
        self.fc = nn.Linear(in_features=hidden_size, out_features=out_size).to(device=device)

    def forward(self, feature_input: torch.Tensor) -> torch.Tensor:
        """ The forward function of MLP Net.

        :param feature_input: input feature, shape=(bs, time_steps, input_size)

        returns: output: the prediction, which is a tensor of shape (bs, 1)

        """

        # ---- Reshape ---- #
        bs = feature_input.shape[0]
        x = feature_input.reshape(bs, -1)  # shape=(bs, time_steps * input_size)

        # ---- Forward computing ---- #
        x = self.mlp(x)  # shape=(bs, hidden_size)
        output = self.fc(x)  # shape=(bs, 1)

        # ---- Return the result ---- #
        return output


if __name__ == "__main__":  # A demo using MLP_Net
    bath_size, time_steps, feature_dim = 64, 1, 26
    hidden_size = 64
    feature = torch.ones((bath_size, time_steps, feature_dim))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_Net(input_size=time_steps * feature_dim, hidden_size=64, device=dev)
    out = model(feature)
    print(out.shape)
