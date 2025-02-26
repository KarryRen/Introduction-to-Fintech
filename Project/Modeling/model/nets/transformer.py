# -*- coding: utf-8 -*-
# @Time    : 2024/4/15 23:26
# @Author  : Karry Ren

""" The Comparison Methods 3: Transformer.

Ref. https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_transformer.py#L258

"""

import logging
import torch
from torch import nn
from typing import Dict
import math


class PositionalEncoding(nn.Module):
    """ The Positional Encoding of Transformer. """

    def __init__(self, d_model: int, max_len: int = 1000):
        """ The init function of PositionalEncoding.

        :param d_model: the model dim
        :param max_len: the max position length

        """

        super(PositionalEncoding, self).__init__()

        # ---- Construct the fix pe (all zero) ---- #
        pe = torch.zeros(max_len, d_model)

        # ---- Computing the positional encoding data ---- #
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape=(max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # shape=(d_model // 2)
        pe[:, 0::2] = torch.sin(position * div_term)  # the even feature, shape=(max_len, d_model//w)
        pe[:, 1::2] = torch.cos(position * div_term)  # the odd feature, shape=(max_len, d_model//w)

        # ---- Add the bs dim ---- #
        pe = pe.unsqueeze(0).transpose(0, 1)
        print(pe.shape)

        # ---- Step 4. Dump to the param ---- #
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ The forward function of PositionalEncoding.

        :param x: the feature need to do PE, shape=(T, bs, d_model)

        Attention: you should be careful about the feature dim, `T` is first !!!
        """
        return x + self.pe[:x.size(0), :]


class Transformer_Net(nn.Module):
    """ The 2 Layer Transformer. model dimension=64. """

    def __init__(
            self, d_feat: int, d_model: int = 64, n_head: int = 4, out_size: int = 3, num_layers: int = 1,
            dropout: float = 0.0, device=torch.device("cpu")
    ):
        """ The init function of Transformer Net.

        :param d_feat: input dim of each step (input size)
        :param d_model: model dim (hidden size)
        :param n_head: the number of head for multi_head_attention
        :param dropout: the dropout ratio
        :param device: the computing device

        """

        super(Transformer_Net, self).__init__()
        self.device = device

        # ---- Log the info of Transformer ---- #
        logging.info(
            f"|||| Using Transformer Now ! d_feat={d_feat}, d_model={d_model}, "
            f"n_head={n_head}, num_layers={num_layers}, dropout_ratio={dropout}||||"
        )

        # ---- Part 1. Linear transformation layer ---- #
        self.linear_layer = nn.Linear(d_feat, d_model).to(device=device)

        # ---- Part 2. Positional Encoding ---- #
        self.pos_encoder = PositionalEncoding(d_model).to(device=device)

        # ---- Part 3. Transformer Encoder ---- #
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout).to(device=device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).to(device=device)

        # ---- Part 4. The output fully connect layer ---- #
        self.fc = nn.Linear(d_model, out_size).to(device=device)

    def forward(self, feature_input: torch.Tensor) -> torch.Tensor:
        """ The forward function of MLP Net.

        :param feature_input: input feature, shape=(bs, time_steps, d_feat)

        returns: output: the prediction, which is a tensor of shape (bs, 1)

        """

        # ---- Transformer Encoding ---- #
        # - transpose from (bs, T, d_feat) to (T, bs, d_feat)
        x = feature_input.transpose(1, 0)  # not batch first
        # - linear transformation
        x = self.linear_layer(x)  # shape=(T, bs, d_model)
        # - positional encoding
        x = self.pos_encoder(x)  # shape=(T, bs, d_model)
        # - transformer encoding
        x = self.transformer_encoder(x)  # shape=(T, bs, d_model)
        # - transpose back, from (T, bs, d_model) to (bs, T, d_model)
        x = x.transpose(1, 0)  # batch first

        # ---- FC to get the prediction ---- #
        # get the last step transformer feature of x
        last_step_trans_x = x[:, -1, :]  # shape=(bs, d_model)
        # use the last step to predict
        output = self.fc(last_step_trans_x)  # shape=(bs, 1)

        # ---- Return the output ---- #
        return output


if __name__ == "__main__":  # A demo using MLP_Net
    bath_size, time_steps, feature_dim = 64, 1, 167
    hidden_size = 64
    feature = torch.ones((bath_size, time_steps, feature_dim))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer_Net(d_feat=167, d_model=128, device=dev)
    out = model(feature)
    print(out.shape)
