# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : Karry Ren

""" Concat data of all Codes. """

import pandas as pd
import os

# ---- Get the data file ---- #
daily_freq_trading_data_file_list = sorted(os.listdir("../../../../Data/daily_trading_factors/raw_data/"))
concat_df = pd.DataFrame()

# ---- For loop to concat ---- #
for i, data_file in enumerate(daily_freq_trading_data_file_list):
    daily_freq_trading_df = pd.read_csv(f"../../../../Data/daily_trading_factors/raw_data/{data_file}")
    concat_df = pd.concat([concat_df, daily_freq_trading_df])
    print(f"finish reading: {i}, {data_file}")

print(concat_df)
concat_df.to_csv("../../../../Data/daily_trading_factors/processed_factors/daily_trading_factors.csv", index=False)
