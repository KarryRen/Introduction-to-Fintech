# -*- coding: utf-8 -*-
# @Time    : 2024/10/20 10:58
# @Author  : Karry Ren

""" The dataset of factors. """

import numpy as np

data_root_path = "../../../Data"
lag_1_data = np.load(f"{data_root_path}/lag_1_factor_data.npz")
print(lag_1_data["feature"][:5], lag_1_data["label"][:5])
lag_2_data = np.load(f"{data_root_path}/lag_2_factor_data.npz")
print(lag_2_data["feature"][:5], lag_2_data["label"][:5])
lag_3_data = np.load(f"{data_root_path}/lag_3_factor_data.npz")
print(lag_3_data["feature"][:5], lag_3_data["label"][:5])
