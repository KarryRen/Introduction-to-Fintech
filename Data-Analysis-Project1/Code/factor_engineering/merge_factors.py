# -*- coding: utf-8 -*-
# @Time    : 2024/10/19 15:59
# @Author  : Karry Ren

""" Merge factors together. """

import pandas as pd

column_list = [
    # the basic columns
    "Date", "Code", "Label",
    # the daily trading factors
    "Open", "Close", "High", "Low", "Volume", "Amount", "Amplitude", "RF_Rate", "RF_Amt", "Turnover",
    "Vwap", "Return_10D", "Return_20D", "Wgt_Return_10D", "Wgt_Return_20D", "Turnover_10D", "Turnover_20D",
    "Std_Turnover_10D", "Std_Turnover_20D", "Std_R_10D", "Std_R_20D", "High_R_Std_20D", "Low_R_Std_20D",
    "Hpl_R_Std_20D", "Hml_R_Std_20D", "Alpha101",
    # the high freq factors

]
data_root_path = "../../../Data"

# ---- Read the daily trading factors ---- #
dt_factors_df = pd.read_csv(f"{data_root_path}/daily_trading_factors/processed_factors/trading_factors.csv")
merged_factors_df = dt_factors_df[column_list]

# ---- Read the high frequency factors ---- #
merged_factors_df.to_pickle(f"{data_root_path}/raw_factors_data.pkl")
print(merged_factors_df)
