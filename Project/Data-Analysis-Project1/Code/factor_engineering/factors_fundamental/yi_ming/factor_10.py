# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : YiMing Jiang

import pandas as pd
import numpy as np
from util import if_end_of_quarter
from util import format_reform

# 10. cashspr
# 季度频率。现金生产率，定义为季度末市值加上长期债务减去总资产除以现金和等价物。

df1 = pd.read_csv("./data/相对价值指标.csv", usecols=["股票代码", "统计截止日期", "市值A"])
df1 = df1[df1["统计截止日期"].apply(lambda x: if_end_of_quarter(x))]
df1.rename({"股票代码": "证券代码"}, axis="columns", inplace=True)

df2 = pd.read_csv("./data/资产负债表.csv", usecols=["证券代码", "报表类型", "统计截止日期", "长期负债合计", "资产总计"], low_memory=False)
df2 = format_reform(df2)

df3 = pd.read_csv("./data/现金流量表(直接法).csv", usecols=["证券代码", "报表类型", "统计截止日期", "期末现金及现金等价物余额"], low_memory=False)
df3 = format_reform(df3)

df4 = pd.merge(df2, df1, on=["证券代码", "统计截止日期"], how="left")
df5 = pd.merge(df4, df3, on=["证券代码", "统计截止日期"], how="left")
df5["cashspr"] = (df5["市值A"] + df5["长期负债合计"] - df5["资产总计"]) / df5["期末现金及现金等价物余额"].replace(0, np.nan)
df5.rename({"cashspr": "Factor_10"}, axis="columns", inplace=True)

df6 = df5[["证券代码", "统计截止日期", "Factor_10"]]
df6.to_csv("./factor/Factor_10.csv", index=False, encoding="utf-8-sig")
