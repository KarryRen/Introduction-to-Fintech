# -*- coding: utf-8 -*-
# @Time    : 2024/10/18 18:03
# @Author  : Karry Ren

""" Compute the daily alpha factors. """

import pandas as pd

# ---- Reading the daily trading csv file ---- #
dt_df_all = pd.read_csv("../../../../Data/daily_trading_factors/processed_factors/raw_daily_trading_values.csv")
dt_df_merged = dt_df_all.copy()
dt_df_all = dt_df_all.pivot(index="Date", columns="Code")  # do the pivot
factor_list = [
    "Open", "Close", "High", "Low", "Volume", "Amount", "Amplitude", "RF_Rate", "RF_Amt", "Turnover",
    "Vwap", "Return_10D", "Return_20D", "Wgt_Return_10D", "Wgt_Return_20D", "Turnover_10D", "Turnover_20D",
    "Std_Turnover_10D", "Std_Turnover_20D", "Std_R_10D", "Std_R_20D", "High_R_Std_20D", "Low_R_Std_20D",
    "Hpl_R_Std_20D", "Hml_R_Std_20D", "Alpha101"
]

# ---- Compute Factors ---- #
# Vwap
vwap_df = pd.DataFrame((dt_df_all["Amount"] / dt_df_all["Volume"]).unstack())  # compute the `Vwap`
vwap_df = vwap_df.rename(columns={0: "Vwap"})  # rename column
dt_df_merged = pd.merge(dt_df_merged, vwap_df, how="inner", on=["Code", "Date"])
print("finish Vwap.")

# Momentum Factors
# - Return 10 Days
return_10d = (dt_df_all["Close"] - dt_df_all["Close"].shift(10)) / dt_df_all["Close"].shift(10)
return_10d_df = pd.DataFrame(return_10d.unstack())
return_10d_df = return_10d_df.rename(columns={0: "Return_10D"})  # rename column
dt_df_merged = pd.merge(dt_df_merged, return_10d_df, how="inner", on=["Code", "Date"])
# - Return 20 Days
return_20d = (dt_df_all["Close"] - dt_df_all["Close"].shift(20)) / dt_df_all["Close"].shift(20)
return_20d_df = pd.DataFrame(return_20d.unstack())
return_20d_df = return_20d_df.rename(columns={0: "Return_20D"})  # rename column
dt_df_merged = pd.merge(dt_df_merged, return_20d_df, how="inner", on=["Code", "Date"])
# - WGT Return 10 Days
return_1d = (dt_df_all["Close"] - dt_df_all["Close"].shift(1)) / dt_df_all["Close"].shift(1)
wgt_return_10d = (return_1d * dt_df_all["Turnover"]).rolling(10).sum() / dt_df_all["Turnover"].rolling(10).sum()
wgt_return_10d_df = pd.DataFrame(wgt_return_10d.unstack())
wgt_return_10d_df = wgt_return_10d_df.rename(columns={0: "Wgt_Return_10D"})  # rename column
dt_df_merged = pd.merge(dt_df_merged, wgt_return_10d_df, how="inner", on=["Code", "Date"])
# - WGT Return 20 Days
return_1d = (dt_df_all["Close"] - dt_df_all["Close"].shift(1)) / dt_df_all["Close"].shift(1)
wgt_return_20d = (return_1d * dt_df_all["Turnover"]).rolling(20).sum() / dt_df_all["Turnover"].rolling(20).sum()
wgt_return_20d_df = pd.DataFrame(wgt_return_20d.unstack())
wgt_return_20d_df = wgt_return_20d_df.rename(columns={0: "Wgt_Return_20D"})  # rename column
dt_df_merged = pd.merge(dt_df_merged, wgt_return_20d_df, how="inner", on=["Code", "Date"])
print("finish Momentum Factors.")

# Turnover Factors
# - Turnover 10 Days
turnover_10d = dt_df_all["Turnover"].rolling(10).mean()
turnover_10d_df = pd.DataFrame(turnover_10d.unstack())
turnover_10d_df = turnover_10d_df.rename(columns={0: "Turnover_10D"})  # rename column
dt_df_merged = pd.merge(dt_df_merged, turnover_10d_df, how="inner", on=["Code", "Date"])
# - Turnover 20 Days
turnover_20d = dt_df_all["Turnover"].rolling(20).mean()
turnover_20d_df = pd.DataFrame(turnover_20d.unstack())
turnover_20d_df = turnover_20d_df.rename(columns={0: "Turnover_20D"})  # rename column
dt_df_merged = pd.merge(dt_df_merged, turnover_20d_df, how="inner", on=["Code", "Date"])
# - Std Turnover 10 Days
std_turnover_10d = dt_df_all["Turnover"].rolling(10).std()
std_turnover_10d_df = pd.DataFrame(std_turnover_10d.unstack())
std_turnover_10d_df = std_turnover_10d_df.rename(columns={0: "Std_Turnover_10D"})  # rename column
dt_df_merged = pd.merge(dt_df_merged, std_turnover_10d_df, how="inner", on=["Code", "Date"])
# - Std Turnover 20 Days
std_turnover_20d = dt_df_all["Turnover"].rolling(20).std()
std_turnover_20d_df = pd.DataFrame(std_turnover_20d.unstack())
std_turnover_20d_df = std_turnover_20d_df.rename(columns={0: "Std_Turnover_20D"})  # rename column
dt_df_merged = pd.merge(dt_df_merged, std_turnover_20d_df, how="inner", on=["Code", "Date"])
print("finish Turnover Factors.")

# Volatility Factors
# - Std R 10 Days
return_1d = (dt_df_all["Close"] - dt_df_all["Close"].shift(1)) / dt_df_all["Close"].shift(1)
std_r_10d_df = pd.DataFrame(return_1d.rolling(10).std().unstack())
std_r_10d_df = std_r_10d_df.rename(columns={0: "Std_R_10D"})
dt_df_merged = pd.merge(dt_df_merged, std_r_10d_df, how="inner", on=["Code", "Date"])
# - Std R 20 Days
return_1d = (dt_df_all["Close"] - dt_df_all["Close"].shift(1)) / dt_df_all["Close"].shift(1)
std_r_20d_df = pd.DataFrame(return_1d.rolling(20).std().unstack())
std_r_20d_df = std_r_20d_df.rename(columns={0: "Std_R_20D"})
dt_df_merged = pd.merge(dt_df_merged, std_r_20d_df, how="inner", on=["Code", "Date"])
# - High R Std 20 Days
high_r = dt_df_all["High"] / dt_df_all["Close"].shift(1)
high_r_20d_df = pd.DataFrame(high_r.rolling(20).std().unstack())
high_r_20d_df = high_r_20d_df.rename(columns={0: "High_R_Std_20D"})
dt_df_merged = pd.merge(dt_df_merged, high_r_20d_df, how="inner", on=["Code", "Date"])
# - Low R Std 20 Days
low_r = dt_df_all["Low"] / dt_df_all["Close"].shift(1)
low_r_20d_df = pd.DataFrame(low_r.rolling(20).std().unstack())
low_r_20d_df = low_r_20d_df.rename(columns={0: "Low_R_Std_20D"})
dt_df_merged = pd.merge(dt_df_merged, low_r_20d_df, how="inner", on=["Code", "Date"])
# - Hml R Std 20D
hml_r_std_20d_df = pd.DataFrame(high_r_20d_df["High_R_Std_20D"] - low_r_20d_df["Low_R_Std_20D"])
hml_r_std_20d_df = hml_r_std_20d_df.rename(columns={0: "Hml_R_Std_20D"})
dt_df_merged = pd.merge(dt_df_merged, hml_r_std_20d_df, how="inner", on=["Code", "Date"])
# - Hpl R Std 20D
hpl_r_std_20d_df = pd.DataFrame(high_r_20d_df["High_R_Std_20D"] + low_r_20d_df["Low_R_Std_20D"])
hpl_r_std_20d_df = hpl_r_std_20d_df.rename(columns={0: "Hpl_R_Std_20D"})
dt_df_merged = pd.merge(dt_df_merged, hpl_r_std_20d_df, how="inner", on=["Code", "Date"])
print("finish Volatility Factors.")

# Alpha 101 Ref https://zhuanlan.zhihu.com/p/28440433
alpha101 = (dt_df_all["Close"] - dt_df_all["Open"]) / (dt_df_all["High"] - dt_df_all["Low"] + 0.001)
alpha101_df = pd.DataFrame(alpha101.unstack())
alpha101_df = alpha101_df.rename(columns={0: "Alpha101"})
dt_df_merged = pd.merge(dt_df_merged, alpha101_df, how="inner", on=["Code", "Date"])
print("finish Alpha101.")

# ---- Compute the Label ---- #
label = (dt_df_all["Open"].shift(-2) - dt_df_all["Open"].shift(-1)) / dt_df_all["Open"].shift(-1)
label_df = pd.DataFrame(label.unstack())
label_df = label_df.rename(columns={0: "Label"})
dt_df_merged = pd.merge(dt_df_merged, label_df, how="inner", on=["Code", "Date"])

# ---- Change dtype and save ---- #
dt_df_merged[["Code", "Date"]] = dt_df_merged[["Code", "Date"]].astype("str")
dt_df_merged[factor_list] = dt_df_merged[factor_list].astype("float32")
dt_df_merged.to_csv("../../../../Data/daily_trading_factors/processed_factors/trading_factors.csv", index=False)
print(dt_df_merged)
