# -*- coding: utf-8 -*-
# @Time    : 2024/10/10 17:15
# @Author  : Karry Ren

""" Reorganize the data to 10 Factors:
    - Open
    - Close
    - High
    - Low
    - Volume
    - Amount
    - Amplitude
    - RF_Rate
    - RF_Amt
    - Turnover
"""

import os
import pandas as pd

# ---- Step 1. Rate the trading dates df---- #
trading_dates_df = pd.read_csv("../../../Data/trading_dates.csv")
trading_dates_df = trading_dates_df.rename(columns={"trade_date": "Date"})

# ---- Step 2. Build up 10 empty factors df ---- #
open_df = trading_dates_df
close_df = trading_dates_df
high_df = trading_dates_df
low_df = trading_dates_df
vol_df = trading_dates_df
amt_df = trading_dates_df
amp_df = trading_dates_df
rf_rate_df = trading_dates_df
rf_amt_df = trading_dates_df
turnover = trading_dates_df

# ---- Step 3. For loop to read and concat ---- #
daily_freq_trading_data_file_list = sorted(os.listdir("../../../../Data/daily_trading_factors/raw_data/"))
for i, data_file in enumerate(daily_freq_trading_data_file_list):
    daily_freq_trading_df = pd.read_csv(f"../../../../Data/daily_trading_factors/raw_data/{data_file}")
    open_df = pd.merge(open_df, daily_freq_trading_df[["Date", "Open"]], how="outer", on="Date").rename(columns={"Open": data_file[:-4]})
    close_df = pd.merge(close_df, daily_freq_trading_df[["Date", "Close"]], how="outer", on="Date").rename(columns={"Close": data_file[:-4]})
    high_df = pd.merge(high_df, daily_freq_trading_df[["Date", "High"]], how="outer", on="Date").rename(columns={"High": data_file[:-4]})
    low_df = pd.merge(low_df, daily_freq_trading_df[["Date", "Low"]], how="outer", on="Date").rename(columns={"Low": data_file[:-4]})
    vol_df = pd.merge(vol_df, daily_freq_trading_df[["Date", "Volume"]], how="outer", on="Date").rename(columns={"Volume": data_file[:-4]})
    amt_df = pd.merge(amt_df, daily_freq_trading_df[["Date", "Amount"]], how="outer", on="Date").rename(columns={"Amount": data_file[:-4]})
    amp_df = pd.merge(amp_df, daily_freq_trading_df[["Date", "Amplitude"]], how="outer", on="Date").rename(columns={"Amplitude": data_file[:-4]})
    rf_rate_df = pd.merge(rf_rate_df, daily_freq_trading_df[["Date", "RF_Rate"]], how="outer", on="Date").rename(columns={"RF_Rate": data_file[:-4]})
    rf_amt_df = pd.merge(rf_amt_df, daily_freq_trading_df[["Date", "RF_Amt"]], how="outer", on="Date").rename(columns={"RF_Amt": data_file[:-4]})
    turnover = pd.merge(turnover, daily_freq_trading_df[["Date", "Turnover"]], how="outer", on="Date").rename(columns={"Turnover": data_file[:-4]})
    print(f"finish: {i}, {data_file}")

# ---- Step 4. Save to the csv ---- #
open_df.to_csv("../../../../Data/daily_trading_factors/reorganized_factors/Open.csv", index=False)
close_df.to_csv("../../../../Data/daily_trading_factors/reorganized_factors/Close.csv", index=False)
high_df.to_csv("../../../../Data/daily_trading_factors/reorganized_factors/High.csv", index=False)
low_df.to_csv("../../../../Data/daily_trading_factors/reorganized_factors/Low.csv", index=False)
vol_df.to_csv("../../../../Data/daily_trading_factors/reorganized_factors/Volume.csv", index=False)
amt_df.to_csv("../../../../Data/daily_trading_factors/reorganized_factors/Amount.csv", index=False)
amp_df.to_csv("../../../../Data/daily_trading_factors/reorganized_factors/Amplitude.csv", index=False)
rf_rate_df.to_csv("../../../../Data/daily_trading_factors/reorganized_factors/RF_Rate.csv", index=False)
rf_amt_df.to_csv("../../../../Data/daily_trading_factors/reorganized_factors/RF_Amt.csv", index=False)
turnover.to_csv("../../../../Data/daily_trading_factors/reorganized_factors/Turnover.csv", index=False)
