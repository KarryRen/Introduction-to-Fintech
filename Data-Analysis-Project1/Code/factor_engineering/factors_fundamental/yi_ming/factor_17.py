# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : YiMing Jiang

import pandas as pd
import numpy as np
from util import format_reform

# 17. chinv
# 季度频率。存货变动除以总资产。

df1 = pd.read_csv("./data/资产负债表.csv", usecols=["证券代码", "报表类型", "统计截止日期", "存货净额", "资产总计"], low_memory=False)
df1 = format_reform(df1)
df1["存货比"] = df1["存货净额"] / df1["资产总计"].replace(0, np.nan)
df1["chinv"] = df1["存货比"] - df1.groupby("证券代码").shift(1)["存货比"]
df1.rename({"chinv": "Factor_17"}, axis="columns", inplace=True)

df2 = df1[["证券代码", "统计截止日期", "Factor_17"]]
df2.to_csv("./factor/Factor_17.csv", index=False, encoding="utf-8-sig")
