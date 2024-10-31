# -*- coding: utf-8 -*-
# @Time    : 2024/10/19 17:00
# @Author  : Karry Ren

""" Select and normalize the data. """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_root_path = "../../../Data"

# ---- Read the data & do visualization ---- #
# read
factors_df = pd.read_pickle(f"{data_root_path}/all_factors.pkl")  # length = 606475
# visual
nan_num = factors_df.isnull().sum(axis=1) / len(factors_df.columns)
plt.figure(dpi=300)
sns.distplot(np.array(nan_num), hist=True, kde=True, color="b")
plt.xlabel("Percentage of NaN Factors")
plt.savefig(f"nan_num_percentage.png", bbox_inches="tight")

# ---- Select the feature columns ---- #
all_column_list = list(factors_df.columns)  # get the column list
util_columns = ["Code", "Date", "Label"]
daily_trading_factor_list = all_column_list[2:2 + 26]  # 26 daily trading factors
high_freq_factor_list = all_column_list[29:29 + 64]  # 64 min to daily factors
fund_factor_list = sorted(all_column_list[93:93 + 22] + all_column_list[126:])  # 22 yi_ming's and 44 lan_yang's (66 in total)
macro_factor_list = all_column_list[93 + 22:126]  # 11 macro factors
factors_list = daily_trading_factor_list + high_freq_factor_list + fund_factor_list + macro_factor_list
factors_df = factors_df[util_columns + factors_list]
factors_num = len(factors_list)  # get the total number

# ---- Select the data using nan ---- #
# use the label to select drop where Label is Nan
factors_df = factors_df.dropna(axis=0, subset=["Label"])  # length = 605151
# use the feature to select, the total number of nan in one line should <= thresh
factors_num_thresh = 3 + 0.1 * factors_num  # the threshold of factors number
factors_df = factors_df[factors_df.isnull().sum(axis=1) <= factors_num_thresh]  # length = 449765

# ---- Do the feature normalization (z-score) and Adjust the Label value ---- #
factors_df_normed = factors_df[["Date", "Code", "Label"]]  # the empty normed factors
factors_df_f = factors_df[["Date", "Code"] + factors_list]  # the raw factors
factors_df_f = factors_df_f.pivot(index="Date", columns="Code")
# do the cross-sectional feature norm one by one
for factor in factors_list:
    factor_cross_mean = factors_df_f[factor].mean(axis=1, skipna=True)  # no nan mean
    factor_cross_std = factors_df_f[factor].std(axis=1, skipna=True)  # no nan std
    normed_factor = factors_df_f[factor].sub(factor_cross_mean, axis=0).div(factor_cross_std + 1e-5, axis=0)  # z-score
    normed_factor = normed_factor.fillna(0)  # fill nan
    normed_factor_df = pd.DataFrame(normed_factor.unstack()).rename(columns={0: factor})  # get the dataframe
    factors_df_normed = pd.merge(factors_df_normed, normed_factor_df, how="inner", on=["Code", "Date"])  # merged
    print(f"Finish factor normalization: `{factor}`.")
# return times 100 and get %
factors_df_normed["Label"] = factors_df_normed["Label"] * 100
assert not factors_df_normed.isnull().values.any(), "There are NaNs in factors_df_normed !!!"

normed_factor_data = np.array(factors_df_normed[factors_list])
print(normed_factor_data.min(), normed_factor_data.max())
sns.distplot(normed_factor_data, hist=True, kde=True, color="b")
plt.xlabel("Factor Value")
plt.savefig(f"normed_factors_data.png", bbox_inches="tight")

# ---- Save it to npz ---- #
stock_list = sorted(factors_df_normed["Code"].unique())
assert len(stock_list) >= 360, "stock code num is not enough !"
for stock in stock_list:
    # --- select stock data
    stock_factors_df_normed = factors_df_normed[factors_df_normed["Code"] == stock]
    # --- do the lag
    # -- do the lag 1
    slag_1_factor_array = np.zeros(shape=(len(stock_factors_df_normed), 1, 1 + len(factors_list)), dtype=np.float32)
    slag_1_factor_array[:, 0, :] = np.array(stock_factors_df_normed[["Label"] + factors_list], dtype=np.float32)
    slag_1_feature, slag_1_label = slag_1_factor_array[:, :, 1:], slag_1_factor_array[:, -1, 0:1]  # split the feature and label
    assert slag_1_feature.dtype == np.float32 and slag_1_label.dtype == np.float32, "dtype ERROR !!"  # check dtype
    # - save to the `.npz` file one by one
    np.savez(f"{data_root_path}/processed_factors/lag_1/{stock}.npz", feature=slag_1_feature, label=slag_1_label)
    print(f"`{stock}`: lag_1 feature and label is saved successfully.")
    # -- do the lag 2
    slag_2_factor_array = np.zeros(shape=(len(stock_factors_df_normed) - 1, 2, 1 + len(factors_list)), dtype=np.float32)
    slag_2_factor_array[:, 0, :] = np.array(stock_factors_df_normed[["Label"] + factors_list], dtype=np.float32)[:-1]
    slag_2_factor_array[:, 1, :] = np.array(stock_factors_df_normed[["Label"] + factors_list], dtype=np.float32)[1:]
    slag_2_feature, slag_2_label = slag_2_factor_array[:, :, 1:], slag_2_factor_array[:, -1, 0:1]  # split the feature and label
    assert slag_2_feature.dtype == np.float32 and slag_2_label.dtype == np.float32, "dtype ERROR !!"  # check dtype
    # - save to the `.npz` file one by one
    np.savez(f"{data_root_path}/processed_factors/lag_2/{stock}.npz", feature=slag_2_feature, label=slag_2_label)
    print(f"`{stock}`: lag_2 feature and label is saved successfully.")
    # --- do the lag 3
    slag_3_factor_array = np.zeros(shape=(len(stock_factors_df_normed) - 2, 3, 1 + len(factors_list)), dtype=np.float32)
    slag_3_factor_array[:, 0, :] = np.array(stock_factors_df_normed[["Label"] + factors_list], dtype=np.float32)[:-2]
    slag_3_factor_array[:, 1, :] = np.array(stock_factors_df_normed[["Label"] + factors_list], dtype=np.float32)[1:-1]
    slag_3_factor_array[:, 2, :] = np.array(stock_factors_df_normed[["Label"] + factors_list], dtype=np.float32)[2:]
    slag_3_feature, slag_3_label = slag_3_factor_array[:, :, 1:], slag_3_factor_array[:, -1, 0:1]  # split the feature and label
    assert slag_3_feature.dtype == np.float32 and slag_3_label.dtype == np.float32, "dtype ERROR !!"  # check dtype
    # - save to the `.npz` file one by one
    np.savez(f"{data_root_path}/processed_factors/lag_3/{stock}.npz", feature=slag_3_feature, label=slag_3_label)
    print(f"`{stock}`: lag_3 feature and label is saved successfully.")
