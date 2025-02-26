# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : YiMing Jiang

import pandas as pd
import numpy as np
from util import format_reform
from util import if_end_of_quarter

# 38. lev
# 季度频率。总负债除以季度末市值。

df1 = pd.read_csv("./data/资产负债表.csv", usecols=["证券代码", "报表类型", "统计截止日期", "负债合计"], low_memory=False)
df1 = format_reform(df1)

df2 = pd.read_csv("./data/相对价值指标.csv", usecols=["股票代码", "统计截止日期", "市值A"])
df2.rename({"股票代码": "证券代码"}, axis="columns", inplace=True)
df2 = df2[df2["统计截止日期"].apply(lambda x: if_end_of_quarter(x))]

df3 = pd.merge(df1, df2, on=["证券代码", "统计截止日期"], how="left")
df3["lev"] = df3["负债合计"] / df3["市值A"].replace(0, np.nan)
df3.rename({"lev": "Factor_38"}, axis="columns", inplace=True)

df4 = df3[["证券代码", "统计截止日期", "Factor_38"]]
df4.to_csv("./factor/Factor_38.csv", index=False, encoding="utf-8-sig")
