# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : YiMing Jiang

import pandas as pd
import numpy as np
from util import format_reform

# 39. lgr
# 季度频率。总负债的季度百分比变化。

df1 = pd.read_csv("./data/资产负债表.csv", usecols=["证券代码", "报表类型", "统计截止日期", "负债合计"], low_memory=False)
df1 = format_reform(df1)
df1["lgr"] = (df1["负债合计"] - df1.groupby("证券代码").shift(1)["负债合计"]) / df1.groupby("证券代码").shift(1)["负债合计"].replace(0, np.nan)
df1.rename({"lgr": "Factor_39"}, axis="columns", inplace=True)

df2 = df1[["证券代码", "统计截止日期", "Factor_39"]]
df2.to_csv("./factor/Factor_39.csv", index=False, encoding="utf-8-sig")
