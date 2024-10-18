# -*- coding: utf-8 -*-
# @Time    : 2024/10/18 17:40
# @Author  : Karry Ren

""" Summary the stock codes. """

import os
import pandas as pd

# ---- Read all files name and separate it ---- #
stock_code_files = sorted(os.listdir(f"../../../Data/daily_trading_factors/raw_data/"))
sz_stock_codes, sh_stock_codes = [], []
for stock_code_file in stock_code_files:
    stock_code = stock_code_file[:-4]
    if stock_code.endswith(".sz"):
        sz_stock_codes.append(stock_code)
    elif stock_code.endswith(".sh"):
        sh_stock_codes.append(stock_code)
    else:
        raise ValueError

# ---- Concat and save ---- #
stock_codes = sz_stock_codes + sh_stock_codes
print(f"sz stock codes: {len(sz_stock_codes)}")
print(f"sh stock codes: {len(sh_stock_codes)}")
pd.DataFrame(stock_codes, columns=["Codes"]).to_csv("../../../Data/stock_codes.csv", index=False)
