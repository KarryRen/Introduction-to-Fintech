# -*- coding: utf-8 -*-
# @Time    : 2024/10/8 16:37
# @Author  : Karry Ren

""" Collect all daily freq trading data of A-share stocks, from START_DATE to END_DATE. """

import akshare as ak
import pandas as pd

# ---- Step 0. Define the start_date and end_date ---- #
START_DATE = "20100101"
END_DATE = "20240601"

# ---- Step 1. Get the stock of sh and sz ---- #
# - sh
stock_sh_code_df = ak.stock_info_sh_name_code(symbol="主板A股")[["证券代码"]]
stock_sh_code_df = stock_sh_code_df.rename(columns={"证券代码": "Code"})
stock_sh_code_df["Code"] = stock_sh_code_df["Code"].apply(lambda x: str(x) + ".sh")
print(f"Before selection: Total are `{len(stock_sh_code_df)}` in sh !")
# - sz
stock_sz = ak.stock_info_sz_name_code(symbol="A股列表")
stock_sz_code_df = stock_sz[stock_sz["板块"] == "主板"][["A股代码"]]
stock_sz_code_df = stock_sz_code_df.rename(columns={"A股代码": "Code"})
stock_sz_code_df["Code"] = stock_sz_code_df["Code"].apply(lambda x: str(x) + ".sz")
print(f"Before selection: Total are `{len(stock_sz_code_df)}` in sz !")

# ---- Step 2. For loop to select the Code List ---- #
selected_ssh_code_list, selected_ssz_code_list = [], []
# - columns
stock_daily_trading_factor_columns = ["股票代码", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
# - rename
stock_daily_trading_factor_re_columns = {
    "股票代码": "Code", "日期": "Date",
    "开盘": "Open", "收盘": "Close", "最高": "High", "最低": "Low",
    "成交量": "Volume", "成交额": "Amount",
    "振幅": "Amplitude", "涨跌幅": "RF_Rate", "涨跌额": "RF_Amt", "换手率": "Turnover"
}
# - sh
stock_sh_code_list = stock_sh_code_df["Code"].values.tolist()
for i, stock_sh_code in enumerate(stock_sh_code_list):
    stock_sh_daily_hist_df = ak.stock_zh_a_hist(symbol=stock_sh_code[:6], period="daily", start_date=START_DATE, end_date=END_DATE, adjust="hfq")
    if len(stock_sh_daily_hist_df) > 100:
        stock_sh_daily_hist_df["日期"] = stock_sh_daily_hist_df["日期"].apply(lambda x: x.strftime("%Y%m%d"))
        stock_sh_daily_hist_df["股票代码"] = stock_sh_daily_hist_df["股票代码"].apply(lambda x: str(x) + ".sh")
        stock_sh_daily_hist_df = stock_sh_daily_hist_df[stock_daily_trading_factor_columns]  # adjust column sequence
        stock_sh_daily_hist_df = stock_sh_daily_hist_df.rename(columns=stock_daily_trading_factor_re_columns)  # rename
        stock_sh_daily_hist_df.to_csv(f"../../../Data/daily_trading_factors/{stock_sh_code}.csv", index=False)
        print(f"Finish stock {i}: `{stock_sh_code}` !")
        selected_ssh_code_list.append(stock_sh_code)
        break
print(f"Total are `{len(selected_ssh_code_list)}` in sh !")
# - sz
stock_sz_code_list = stock_sz_code_df["Code"].values.tolist()
for i, stock_sz_code in enumerate(stock_sz_code_list):
    stock_sz_daily_hist_df = ak.stock_zh_a_hist(symbol=stock_sz_code[:6], period="daily", start_date=START_DATE, end_date=END_DATE, adjust="hfq")
    if len(stock_sz_daily_hist_df) > 100:
        stock_sz_daily_hist_df["日期"] = stock_sz_daily_hist_df["日期"].apply(lambda x: x.strftime("%Y%m%d"))
        stock_sz_daily_hist_df["股票代码"] = stock_sz_daily_hist_df["股票代码"].apply(lambda x: str(x) + ".sh")
        stock_sz_daily_hist_df = stock_sz_daily_hist_df[stock_daily_trading_factor_columns]  # adjust column sequence
        stock_sz_daily_hist_df = stock_sz_daily_hist_df.rename(columns=stock_daily_trading_factor_re_columns)  # rename
        stock_sz_daily_hist_df.to_csv(f"../../../Data/daily_trading_factors/{stock_sz_code}.csv", index=False)
        print(f"Finish stock {i}: `{stock_sz_code}` !")
        selected_ssz_code_list.append(stock_sz_code)
        break
print(f"Total are `{len(selected_ssz_code_list)}` in sz !")
# save code
pd.DataFrame(selected_ssh_code_list + selected_ssz_code_list, columns=["Code"]).to_csv("../../../Data/stock_code.csv", index=False)
