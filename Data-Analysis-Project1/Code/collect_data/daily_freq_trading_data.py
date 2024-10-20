# -*- coding: utf-8 -*-
# @Time    : 2024/10/8 16:37
# @Author  : Karry Ren

""" Collect all daily freq trading data of stocks in stock list, from START_DATE to END_DATE. """

import akshare as ak
import pandas as pd

# ---- Define the start_date and end_date ---- #
START_DATE, END_DATE = "20190101", "20240601"

# ---- Get the stock code list ---- #
stock_codes_df = pd.read_csv("../../../Data/zz500_stocks_240930.csv")
stock_code_list = stock_codes_df["Code"].apply(lambda x: str(x).lower())

# ---- For loop to select the Code List ---- #
# - columns
stock_daily_trading_factor_columns = [
    "股票代码", "日期", "开盘", "收盘", "最高", "最低",
    "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"
]
# - rename columns
stock_daily_trading_factor_re_columns = {
    "股票代码": "Code", "日期": "Date",
    "开盘": "Open", "收盘": "Close", "最高": "High", "最低": "Low",
    "成交量": "Volume", "成交额": "Amount", "振幅": "Amplitude",
    "涨跌幅": "RF_Rate", "涨跌额": "RF_Amt", "换手率": "Turnover"
}
# - for loop to select
for i, stock_code in enumerate(stock_code_list):
    stock_daily_hist_df = ak.stock_zh_a_hist(symbol=stock_code[:6], period="daily", start_date=START_DATE, end_date=END_DATE, adjust="hfq")
    if len(stock_daily_hist_df) > 0:
        stock_daily_hist_df["日期"] = stock_daily_hist_df["日期"].apply(lambda x: x.strftime("%Y%m%d"))
        stock_daily_hist_df["股票代码"] = stock_code  # reset the stock code
        stock_daily_hist_df = stock_daily_hist_df[stock_daily_trading_factor_columns]  # adjust column sequence
        stock_daily_hist_df = stock_daily_hist_df.rename(columns=stock_daily_trading_factor_re_columns)  # rename
        stock_daily_hist_df.to_csv(f"../../../Data/daily_trading_factors/raw_data/{stock_code}.csv", index=False)
        print(f"Finish stock {i}: `{stock_code}` !")
    else:
        print(f"Stock {i}: `{stock_code}` No trading dates are in the period of {START_DATE} to {END_DATE} !")
