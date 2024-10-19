# -*- coding: utf-8 -*-
# @Time    : 2024/10/19 17:00
# @Author  : Karry Ren

""" Select and normalize the data. """
import numpy as np
import pandas as pd

data_root_path = "../../../Data"

# ---- Read the data ---- #
factors_df = pd.read_pickle(f"{data_root_path}/raw_factors_data.pkl")
factors_num = len(factors_df.columns) - 3  # get the factors number

# ---- Select the data ---- #
# use the label to select
factors_df = factors_df.dropna(axis=0, subset=["Label"])  # drop where Label is Nan
# use the feature to select
factors_num_thresh = 3 + 0.1 * factors_num  # the threshold of factors number
factors_df = factors_df[factors_df.isnull().sum(axis=1) <= factors_num_thresh]

# ---- Do the normalization (z-score) ---- #
factors_list = [
    "Open", "Close", "High", "Low", "Volume", "Amount", "Amplitude", "RF_Rate", "RF_Amt", "Turnover",
    "Vwap", "Return_10D", "Return_20D", "Wgt_Return_10D", "Wgt_Return_20D", "Turnover_10D", "Turnover_20D",
    "Std_Turnover_10D", "Std_Turnover_20D", "Std_R_10D", "Std_R_20D", "High_R_Std_20D", "Low_R_Std_20D",
    "Hpl_R_Std_20D", "Hml_R_Std_20D", "Alpha101"
]
factors_df_normed, factors_df_f = factors_df[["Date", "Code", "Label"]], factors_df[["Date", "Code"] + factors_list]
# factors_df_f = factors_df_f.pivot(index="Date", columns="Code")
# for factor in factors_list:  # do the cross norm one by one
#     factor_cross_mean = factors_df_f[factor].mean(axis=1, skipna=True)  # no nan mean
#     factor_cross_std = factors_df_f[factor].std(axis=1, skipna=True)  # no nan std
#     normed_factor = factors_df_f[factor].sub(factor_cross_mean, axis=0).div(factor_cross_std + 1e-5, axis=0).fillna(0)  # z-score & fill nan
#     normed_factor_df = pd.DataFrame(normed_factor.unstack()).rename(columns={0: factor})  # get the dataframe
#     factors_df_normed = pd.merge(factors_df_normed, normed_factor_df, how="inner", on=["Code", "Date"])  # merged
#     break

# ---- Check no nan ---- #
assert not factors_df_normed.isnull().values.any(), "There are NaNs in factors_df_normed !!!"

# ---- Save it to npz, factors 和 labels 分开存, 增加可读性 ---- #
# - the lag 1 data
lag_1_factor_array = np.zeros(shape=(len(factors_df_normed), 1, 1))
lag_1_factor_array[:, 0, :] = np.array(factors_df_normed[["Label"]], dtype=np.float32)
# np.save("lag_1_factor_array.npy", lag_1_factor_array)
# - the lag n data
stock_code_list = sorted(factors_df_normed["Code"].unique())
lag_2_factor_array = np.zeros(shape=(len(factors_df_normed) - len(stock_code_list) * 1, 2, 1))
lag_3_factor_array = np.zeros(shape=(len(factors_df_normed) - len(stock_code_list) * 3, 3, 1))
l2_steps, l3_steps = 0, 0
for stock_code in stock_code_list:
    # get the target stock factor array
    stock_factor_array = np.array(factors_df_normed[factors_df_normed["Code"] == stock_code][["Label"]].values)
    for i in range(0, len(stock_factor_array) - 1):
        lag_2_factor_array[l2_steps + i, :] = stock_factor_array[i:i + 2]
        l2_steps += 1
    for i in range(0, len(stock_factor_array) - 2):
        lag_3_factor_array[l3_steps + i, :] = stock_factor_array[i:i + 3]
        l3_steps += 1
