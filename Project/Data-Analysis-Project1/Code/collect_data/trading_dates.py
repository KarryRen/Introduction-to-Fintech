# -*- coding: utf-8 -*-
# @Time    : 2024/10/9 23:02
# @Author  : Karry Ren

""" Get all trading dates of China from START_DATE to END_DATE. """

import akshare as ak
import datetime

# ---- Step 1. Define the start and end date ---- #
START_DATE = datetime.date(2019, 1, 1)
END_DATE = datetime.date(2024, 6, 1)

# ---- Step 1. Get all date dataframe ---- #
trading_dates_df = ak.tool_trade_date_hist_sina()

# ---- Step 2. Select the interval trading date dataframe and transfer the format ---- #
trading_dates_df = trading_dates_df[trading_dates_df["trade_date"] >= START_DATE]
trading_dates_df = trading_dates_df[trading_dates_df["trade_date"] <= END_DATE]
trading_dates_df["trade_date"] = trading_dates_df["trade_date"].apply(lambda x: x.strftime("%Y%m%d"))

# ---- Step 3. Save the data to `.csv` file ---- #
trading_dates_df.to_csv("../../../Data/trading_dates.csv", index=False)
print(f"Save successful, total `{len(trading_dates_df)}` trading dates.")
