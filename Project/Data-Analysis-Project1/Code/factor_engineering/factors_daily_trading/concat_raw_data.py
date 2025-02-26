# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : Karry Ren

""" Concat data of all Codes. """
import numpy as np
import pandas as pd
import os

# ---- Get the data file ---- #
daily_freq_trading_data_file_list = sorted(os.listdir("../../../../Data/daily_trading_factors/raw_data/"))
concat_df = pd.DataFrame()

# ---- For loop to collect ---- #
df_list = []
for i, data_file in enumerate(daily_freq_trading_data_file_list):
    daily_freq_trading_df = pd.read_csv(f"../../../../Data/daily_trading_factors/raw_data/{data_file}")
    df_list.append(daily_freq_trading_df)

# ---- Concat the df ---- #
concat_df = pd.concat(df_list)
print(len(concat_df))
concat_df.to_csv("../../../../Data/daily_trading_factors/processed_factors/raw_daily_trading_values.csv", index=False)

# ---- Description ---- #
trading_dates = []
for df in df_list:
    trading_dates.append(len(df))
print(max(trading_dates), min(trading_dates))
print(daily_freq_trading_data_file_list[np.argmax(trading_dates)], daily_freq_trading_data_file_list[np.min(trading_dates)])
