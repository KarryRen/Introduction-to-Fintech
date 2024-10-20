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

# ---- Select the data using nan ---- #
# use the label to select
factors_df = factors_df.dropna(axis=0, subset=["Label"])  # drop where Label is Nan
# use the feature to select, the total number of nan in one line should <= thresh
factors_num_thresh = 3 + 0.1 * factors_num  # the threshold of factors number
factors_df = factors_df[factors_df.isnull().sum(axis=1) <= factors_num_thresh]

# ---- Do the feature normalization (z-score) and Adjust the Label value ---- #
factors_list = [
    "Open", "Close", "High", "Low", "Volume", "Amount", "Amplitude", "RF_Rate", "RF_Amt", "Turnover",
    "Vwap", "Return_10D", "Return_20D", "Wgt_Return_10D", "Wgt_Return_20D", "Turnover_10D", "Turnover_20D",
    "Std_Turnover_10D", "Std_Turnover_20D", "Std_R_10D", "Std_R_20D", "High_R_Std_20D", "Low_R_Std_20D",
    "Hpl_R_Std_20D", "Hml_R_Std_20D", "Alpha101"
]  # the feature list
factors_df_normed = factors_df[["Date", "Code", "Label"]]  # the empty normed factors
factors_df_f = factors_df[["Date", "Code"] + factors_list]  # the raw factors
factors_df_f = factors_df_f.pivot(index="Date", columns="Code")
# do the cross-sectional feature norm one by one
for factor in factors_list:
    factor_cross_mean = factors_df_f[factor].mean(axis=1, skipna=True)  # no nan mean
    factor_cross_std = factors_df_f[factor].std(axis=1, skipna=True)  # no nan std
    normed_factor = factors_df_f[factor].sub(factor_cross_mean, axis=0).div(factor_cross_std + 1e-5, axis=0).fillna(0)  # z-score & fill nan
    normed_factor_df = pd.DataFrame(normed_factor.unstack()).rename(columns={0: factor})  # get the dataframe
    factors_df_normed = pd.merge(factors_df_normed, normed_factor_df, how="inner", on=["Code", "Date"])  # merged
    print(f"Finish factor normalization: {factor}.")
# return times 100 and get %
factors_df_normed["Label"] = factors_df_normed["Label"] * 100

# ---- Check no nan ---- #
assert not factors_df_normed.isnull().values.any(), "There are NaNs in factors_df_normed !!!"

# ---- Save it to npz ---- #
# - the lag 1 data (easy to build)
lag_1_factor_array = np.zeros(shape=(len(factors_df_normed), 1, 1 + len(factors_list)), dtype=np.float32)
lag_1_factor_array[:, 0, :] = np.array(factors_df_normed[["Label"] + factors_list], dtype=np.float32)  # the lag 1
lag_1_feature, lag_1_label = lag_1_factor_array[:, :, 1:], lag_1_factor_array[:, -1, 0:1]  # split the feature and label
assert lag_1_feature.dtype == np.float32 and lag_1_label.dtype == np.float32, "dtype ERROR !!"  # check dtype
np.savez(f"{data_root_path}/lag_1_factor_data.npz", feature=lag_1_feature, label=lag_1_label)  # save to the `.npz` file
# - the lag n data (you should be careful about the following codes)
stock_code_list = sorted(factors_df_normed["Code"].unique())  # we will lag the data for each code
lag_2_factor_array = np.zeros(shape=(len(factors_df_normed) - len(stock_code_list) * 1, 2, 1 + len(factors_list)), dtype=np.float32)
lag_3_factor_array = np.zeros(shape=(len(factors_df_normed) - len(stock_code_list) * 2, 3, 1 + len(factors_list)), dtype=np.float32)
l2_steps, l3_steps = 0, 0
for s, stock_code in enumerate(stock_code_list):  # because different have different length, only can do the for loop :(
    # get the target stock factor array
    stock_factor_array = np.array(factors_df_normed[factors_df_normed["Code"] == stock_code][["Label"] + factors_list], dtype=np.float32)
    # in forloop to stack
    for i in range(0, len(stock_factor_array) - 1):
        lag_2_factor_array[l2_steps, :] = stock_factor_array[i:i + 2]
        l2_steps += 1
    # in forloop to stack
    for i in range(0, len(stock_factor_array) - 2):
        lag_3_factor_array[l3_steps, :] = stock_factor_array[i:i + 3]
        l3_steps += 1
    print(f"Finish stock stacking {s}: stock_code {stock_code}.")
lag_2_feature, lag_2_label = lag_2_factor_array[:, :, 1:], lag_2_factor_array[:, -1, 0:1]  # split the feature and label
assert lag_2_feature.dtype == np.float32 and lag_2_label.dtype == np.float32, "dtype ERROR !!"  # check dtype
np.savez(f"{data_root_path}/lag_2_factor_data.npz", feature=lag_2_feature, label=lag_2_label)  # save to the `.npz` file
lag_3_feature, lag_3_label = lag_3_factor_array[:, :, 1:], lag_3_factor_array[:, -1, 0:1]  # split the feature and label
assert lag_3_feature.dtype == np.float32 and lag_3_label.dtype == np.float32, "dtype ERROR !!"  # check dtype
np.savez(f"{data_root_path}/lag_3_factor_data.npz", feature=lag_3_feature, label=lag_3_label)  # save to the `.npz` file
