# -*- coding: utf-8 -*-
# @Time    : 2024/10/8 16:37
# @Author  : Karry Ren

""" Collect all trading data of A-share stocks, from 20000101 to 20231231. """

import akshare as ak

# ---- Step 1. Get the stock of sh and sz ---- #
# - sh
stock_sh_code_df = ak.stock_info_sh_name_code(symbol="主板A股")[["证券代码"]]
stock_sh_code_df = stock_sh_code_df.rename(columns={"证券代码": "Code"})
stock_sh_code_df["Code"] = stock_sh_code_df["Code"].apply(lambda x: str(x) + ".sh")
# - sz
stock_sz = ak.stock_info_sz_name_code(symbol="A股列表")
stock_sz_code_df = stock_sz[stock_sz["板块"] == "主板"][["A股代码"]]
stock_sz_code_df = stock_sz_code_df.rename(columns={"A股代码": "Code"})
stock_sz_code_df["Code"] = stock_sz_code_df["Code"].apply(lambda x: str(x) + ".sz")

# ---- Step 2. For loop to select the Code List ---- #
selected_ssh_code_list, selected_ssz_code_list = [], []
stock_daily_trading_factor_columns = ["股票代码", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
# - sh
# stock_sh_code_list = stock_sh_code_df["Code"].values.tolist()
# for stock_sh_code in stock_sh_code_list:
#     stock_sh_daily_hist_df = ak.stock_zh_a_hist(symbol=stock_sh_code[:6], period="daily", start_date="20000101", end_date="20240101", adjust="hfq")
#     if len(stock_sh_daily_hist_df) > 100:
#         stock_sh_daily_hist_df["日期"] = stock_sh_daily_hist_df["日期"].apply(lambda x: x.strftime("%Y%m%d"))
#         stock_sh_daily_hist_df["股票代码"] = stock_sh_daily_hist_df["股票代码"].apply(lambda x: str(x) + ".sh")
#         stock_sh_daily_hist_df = stock_sh_daily_hist_df[stock_daily_trading_factor_columns]  # adjust column sequence
#         stock_sh_daily_hist_df.to_csv(f"../../../Data/daily_trading_factors/{stock_sh_code}.csv", index=False, encoding="utf_8_sig")
#         print(f"Finish stock `{stock_sh_code}` !")
#         selected_ssh_code_list.append(stock_sh_code)
# print(selected_ssh_code_list)
# - sz
stock_sz_code_list = stock_sz_code_df["Code"].values.tolist()
for stock_sz_code in stock_sz_code_list:
    stock_sz_daily_hist_df = ak.stock_zh_a_hist(symbol=stock_sz_code[:6], period="daily", start_date="20000101", end_date="20240101", adjust="hfq")
    if len(stock_sz_daily_hist_df) > 100:
        stock_sz_daily_hist_df["日期"] = stock_sz_daily_hist_df["日期"].apply(lambda x: x.strftime("%Y%m%d"))
        stock_sz_daily_hist_df["股票代码"] = stock_sz_daily_hist_df["股票代码"].apply(lambda x: str(x) + ".sh")
        stock_sz_daily_hist_df = stock_sz_daily_hist_df[stock_daily_trading_factor_columns]  # adjust column sequence
        stock_sz_daily_hist_df.to_csv(f"../../../Data/daily_trading_factors/{stock_sz_code}.csv", index=False, encoding="utf_8_sig")
        print(f"Finish stock `{stock_sz_code}` !")
        selected_ssz_code_list.append(stock_sz_code)
        break
print(selected_ssz_code_list)
