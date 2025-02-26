# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : YiMing Jiang

""" 受限于公司文件，有部分股票不在其中，导致部分股票没有 13, 14 号因子 """

import pandas as pd
import numpy as np
from util import diff
from util import if_end_of_quarter
from util import format_reform

# 13. chato
# 季度频率。销售变化量除以平均总资产。

# 14. chato_ia
# 季度频率。经行业调整的销售变化除以平均总资产。

df1 = pd.read_csv("./data/利润表.csv", usecols=["证券代码", "报表类型", "统计截止日期", "营业总收入"])
df1 = format_reform(df1)
df1 = diff(df1, col=["营业总收入"])
df1["DTSale"] = df1["营业总收入"] - df1.groupby("证券代码").shift(1)["营业总收入"]

df2 = pd.read_csv("./data/资产负债表.csv", usecols=["证券代码", "报表类型", "统计截止日期", "资产总计"], low_memory=False)
df2 = format_reform(df2)
df2["AveTA"] = (df2["资产总计"] + df2.groupby("证券代码").shift(1)["资产总计"]) / 2

df3 = pd.merge(df1, df2, on=["证券代码", "统计截止日期"], how="left")
df3["chato"] = df3["DTSale"] / df3["AveTA"].replace(0, np.nan)

df4 = pd.read_csv("./data/公司文件.csv", usecols=["证券代码", "行业代码C"])

df5 = pd.merge(df3, df4, on=["证券代码"])

df6 = df5[["行业代码C", "统计截止日期", "chato"]].groupby(["行业代码C", "统计截止日期"]).mean().reset_index()
df6.rename(columns={"chato": "chato_i"}, inplace=True)

df7 = pd.merge(df5, df6, on=["行业代码C", "统计截止日期"])
df7["chato_ia"] = df7["chato"] - df7["chato_i"]

# df8 = df7[["证券代码", "统计截止日期", "chato", "chato_ia"]]
# df8.to_csv("../../output/csmar/季_13_14.csv", index=False, encoding="utf-8-sig")

df7.rename({"chato": "Factor_13"}, axis="columns", inplace=True)
df7.rename({"chato_ia": "Factor_14"}, axis="columns", inplace=True)

df_bm = df7[["证券代码", "统计截止日期", "Factor_13"]]
df_bm.sort_values(["证券代码", "统计截止日期"], inplace=True)
df_bm.to_csv("./factor/Factor_13.csv", index=False, encoding="utf-8-sig")

df_bm_ia = df7[["证券代码", "统计截止日期", "Factor_14"]]
df_bm_ia.sort_values(["证券代码", "统计截止日期"], inplace=True)
df_bm_ia.to_csv("./factor/Factor_14.csv", index=False, encoding="utf-8-sig")