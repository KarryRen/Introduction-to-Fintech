# -*- coding: utf-8 -*-
# @Time    : 2024/10/31 16:10
# @Author  : Karry Ren

""" The convolutional model. """

import logging
import torch
from torch import nn


class Conv_Net(nn.Module):
    def __init__(self, hidden_size: int = 64, out_size: int = 3, device: torch.device = torch.device("cpu")):
        """ The init function of Conv Net.

        :param hidden_size: hidden size
        :param device: the computing device

        """

        super(Conv_Net, self).__init__()
        self.device = device

        # ---- Log the info of Conv ---- #
        logging.info(f"|||| Using Conv Model Now !")

        # ---- Build up the model ---- #
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=(1, 4), stride=(1, 4), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size * 2, kernel_size=(1, 4), stride=(1, 4), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size * 2, out_channels=hidden_size, kernel_size=(1, 10), bias=False),
            nn.ReLU()
        ).to(device=device)
        self.fc = nn.Linear(in_features=hidden_size, out_features=out_size).to(device=device)

    def forward(self, feature_input: torch.Tensor) -> torch.Tensor:
        """ The forward function of Conv Net.

        :param feature_input: input feature, shape=(bs, time_steps, input_size)

        returns: output: the prediction, which is a tensor of shape (bs, 1)

        """

        # ---- Reshape ---- #
        bs, ttf = feature_input.shape[0], feature_input.shape[1] * feature_input.shape[2]
        x = feature_input.reshape(bs, 1, 1, ttf)  # shape=(bs, 1, 1, time_steps * input_size)

        # ---- Forward computing ---- #
        x = self.conv(x).reshape(bs, -1)  # shape=(bs, hidden_size)
        output = self.fc(x)  # shape=(bs, 1)

        # ---- Return the result ---- #
        return output


if __name__ == "__main__":  # A demo using Conv_Net
    bath_size, time_steps, feature_dim = 64, 1, 167
    hidden_size = 64
    feature = torch.ones((bath_size, time_steps, feature_dim))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv_Net(hidden_size=64, device=dev)
    out = model(feature)
    print(out.shape)
