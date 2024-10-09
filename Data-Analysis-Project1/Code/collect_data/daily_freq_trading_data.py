# -*- coding: utf-8 -*-
# @Time    : 2024/10/8 16:37
# @Author  : Karry Ren

""" Collect all trading data of A-share stocks, from 20000101 to 20231231. """

import akshare as ak

# print(ak.stock_zh_a_spot_em())  # only can

stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20000101", end_date="20240101", adjust="")
print(stock_zh_a_hist_df.iloc[0, :])
