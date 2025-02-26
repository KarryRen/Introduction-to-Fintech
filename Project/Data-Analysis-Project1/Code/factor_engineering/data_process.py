"""
 Select and normalize the data. just like select_and_norm.py,but functinalized 
"""
import numpy as np
import pandas as pd

data_root_path = "../../../Data"


def get_features_cols(factors_df):
    # ---- Select the feature columns ---- #
    all_column_list = list(factors_df.columns)  # get the columns
    daily_trading_factor_list = all_column_list[2:2 + 26]  # 26 daily trading factors
    high_freq_factor_list = all_column_list[29:29 + 64]  # 64 min to daily factors
    fund_factor_list = sorted(all_column_list[93:93 + 22] + all_column_list[126:])  # 22 yi_ming's and 44 lan_yang's (66 in total)
    macro_factor_list = all_column_list[93 + 22:126]  # 11 macro factors
    factors_list = daily_trading_factor_list + high_freq_factor_list + fund_factor_list + macro_factor_list
    return factors_list


def dropna_row(factors_df, factors_num, threshold_pct=0.1):
    # ---- Select the data using nan ---- #
    # use the label to select drop where Label is Nan
    factors_df = factors_df.dropna(axis=0, subset=["Label"])  # length = 605151
    # use the feature to select, the total number of nan in one line should <= thresh
    factors_num_thresh = 3 + threshold_pct * factors_num  # the threshold of factors number
    factors_df = factors_df[factors_df.isnull().sum(axis=1) <= factors_num_thresh]  # length = 449765
    print("remained data length:", len(factors_df))
    return factors_df


def do_normalization(factors_df, factors_list):
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
    return factors_df_normed


def save_npz_data(factors_df_normed, factors_list):
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


def gen_lag_data(df, lags, dropnan=True):
    """
    generate columns of lagged data to the df 
    """
    n_vars = df.shape[1]
    raw_columns = df.columns
    cols, names = [], []
    # add lag 0 data
    cols.append(df)
    names += [f"{i}" for i in raw_columns]
    # generate data of lag (t-n, ... t-1)
    for i in range(1, lags + 1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (raw_columns[j % n_vars], i)) for j in range(n_vars)]
    # merge all lagged data
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop columns of lagged label
    for i in range(1, lags + 1):
        agg.drop(columns=['Label(t-%d)' % i], inplace=True)
    # eliminate nan row
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# Read the data 
factors_df = pd.read_pickle(f"{data_root_path}/all_factors1.pkl")  # length = 606475

# Select the feature columns
factors_list = get_features_cols(factors_df)
util_columns = ["Code", "Date", "Label"]
factors_df = factors_df[util_columns + factors_list]
factors_num = len(factors_list)

# Select the data using nan
factors_df = dropna_row(factors_df, factors_num, threshold_pct=0.1)

# Do the feature normalization (z-score) and Adjust the Label value 
factors_df_normed = do_normalization(factors_df, factors_list)

# Save it to npz 
save_npz_data(factors_df_normed, factors_list)

# generate lagged data for each stock
# lag1_factors_df = factors_df_normed.groupby("Code").apply(lambda x:gen_lag_data(x.sort_values('Date').set_index(['Date']),lags=1,dropnan=True),include_groups=False)
# lag2_factors_df = factors_df_normed.groupby("Code").apply(lambda x:gen_lag_data(x.sort_values('Date').set_index(['Date']),lags=2,dropnan=True),include_groups=False)
lag3_factors_df = factors_df_normed.groupby("Code").apply(lambda x:gen_lag_data(x.sort_values('Date').set_index(['Date']),lags=3,dropnan=True),include_groups=False)
lag3_factors_df.to_pickle("lag3_factors_df.pkl")
