# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : YiMing Jiang

import numpy as np
import pandas as pd
from util import diff
from util import format_reform
import warnings

warnings.filterwarnings("ignore")

# 01. acc
# 半年频率。遵循Sloan (1996)对应计利润的定义来构建acc：
# acc = [(delta_CA - delta_CASH) - (delta_CL - delta_STD - delta_TP) - Dep]/Total_Assets
# 其中
# delta表示两个连续周期之间的差
# CA = 流动资产
# CASH = 现金及其等价物
# CL = 流动负债
# STD = 包括在流动负债中的债务
# TP = 应付所得税
# Dep = 折旧及摊销费用

# 02. absacc
# 半年频率。acc的绝对值。

# 60. pctacc
# 半年频率。与acc相同，不同的是分子要除以净收入的绝对值；如果净收入= 0，则净收入设置为0.01作为分母。

# 81. stdacc
# 季度频率（实际计算时改为半年度频率）。16个季度的应计收益的标准差（从第t-16月到第t-1月）。


df1 = pd.read_csv("./data/资产负债表.csv",
                  usecols=["证券代码", "报表类型", "统计截止日期", "现金及存放中央银行款项", "流动资产合计", "资产总计", "短期借款", "应交税费",
                           "流动负债合计"], low_memory=False)
df1 = format_reform(df1)
df1["现金及存放中央银行款项"] = df1["现金及存放中央银行款项"].fillna(0)
df1["短期借款"] = df1["短期借款"].fillna(0)

df2 = pd.read_csv("./data/利润表.csv", usecols=["证券代码", "报表类型", "统计截止日期", "净利润"])
df2 = format_reform(df2)
df2["证券代码"] = df2["证券代码"].astype(int)

df3 = pd.read_csv("./data/现金流量表(间接法).csv",
                  usecols=["证券代码", "报表类型", "统计截止日期", "固定资产折旧、油气资产折耗、生产性生物资产折旧", "无形资产摊销",
                           "长期待摊费用摊销"], low_memory=False)
df3 = format_reform(df3)
df3["固定资产折旧、油气资产折耗、生产性生物资产折旧"] = df3["固定资产折旧、油气资产折耗、生产性生物资产折旧"].fillna(0)
df3["无形资产摊销"] = df3["无形资产摊销"].fillna(0)
df3["长期待摊费用摊销"] = df3["长期待摊费用摊销"].fillna(0)

df4 = pd.merge(pd.merge(df1, df2, on=["证券代码", "统计截止日期"], how="left"), df3, on=["证券代码", "统计截止日期"], how="left")
df4["半年"] = df4["统计截止日期"].apply(lambda x: True if x[5:7] in ["06", "12"] else False)

df5 = df4[df4["半年"]].reset_index().drop(columns=["index"])
df5 = diff(df5, col=["净利润", "固定资产折旧、油气资产折耗、生产性生物资产折旧", "无形资产摊销", "长期待摊费用摊销"], freq="半年")
df5["折旧摊销"] = df5["固定资产折旧、油气资产折耗、生产性生物资产折旧"] + df5["无形资产摊销"] + df5["长期待摊费用摊销"]
df5["delta_流动资产合计"] = df5["流动资产合计"] - df5.groupby("证券代码").shift(1)["流动资产合计"]
df5["delta_现金及存放中央银行款项"] = df5["现金及存放中央银行款项"] - df5.groupby("证券代码").shift(1)["现金及存放中央银行款项"]
df5["delta_流动负债合计"] = df5["流动负债合计"] - df5.groupby("证券代码").shift(1)["流动负债合计"]
df5["delta_短期借款"] = df5["短期借款"] - df5.groupby("证券代码").shift(1)["短期借款"]
df5["delta_应交税费"] = df5["应交税费"] - df5.groupby("证券代码").shift(1)["应交税费"]
df5["acc"] = ((df5["delta_流动资产合计"] - df5["delta_现金及存放中央银行款项"]) - (
        df5["delta_流动负债合计"] - df5["delta_短期借款"] - df5["delta_应交税费"]) - df5["折旧摊销"]) / df5["资产总计"].replace(0, np.nan)
df5["absacc"] = abs(df5["acc"])
df5["pctacc"] = df5["acc"] * df5["资产总计"] / df5["净利润"]
df5["stdacc"] = df5[["证券代码", "acc"]].groupby("证券代码").rolling(3).std().reset_index()["acc"]
df5.rename({"acc": "Factor_1", "absacc": "Factor_2", "pctacc": "Factor_60", "stdacc": "Factor_81"}, axis="columns", inplace=True)

df_acc = df5[["证券代码", "统计截止日期", "Factor_1"]]
df_acc.to_csv("./factor/Factor_1.csv", index=False, encoding="utf-8-sig")

df_absacc = df5[["证券代码", "统计截止日期", "Factor_2"]]
df_absacc.to_csv("./factor/Factor_2.csv", index=False, encoding="utf-8-sig")

df_pctacc = df5[["证券代码", "统计截止日期", "Factor_60"]]
df_pctacc.to_csv("./factor/Factor_60.csv", index=False, encoding="utf-8-sig")

df_stdacc = df5[["证券代码", "统计截止日期", "Factor_81"]]
df_stdacc.to_csv("./factor/Factor_81.csv", index=False, encoding="utf-8-sig")
