# -*- coding: utf-8 -*-
# @Time    : 2024/10/8 16:37
# @Author  : Karry Ren

""" Collect all trading data of A-share stocks, from 20000101 to 20231231. """

import akshare as ak

# ---- Step 1. Get the stock of sh and sz ---- #
# - sh
stock_sh_code_df = ak.stock_info_sh_name_code(symbol="主板A股")["证券代码"]
stock_sh_code_df[""]
# - sz
stock_sz = ak.stock_info_sz_name_code(symbol="A股列表")
stock_sz = stock_sz[stock_sz["板块"] == "主板"]["A股代码"]
print(len(stock_sz))

# stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20000101", end_date="20240101", adjust="")
# print(stock_zh_a_hist_df.iloc[0, :])
