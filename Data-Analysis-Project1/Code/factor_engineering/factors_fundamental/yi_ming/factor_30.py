# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : YiMing Jiang

import pandas as pd
import numpy as np
from util import format_reform

# 30. egr
# 季度频率。股本账面价值的季度变动百分比。

df1 = pd.read_csv("./data/资产负债表.csv", usecols=["证券代码", "报表类型", "统计截止日期", "所有者权益合计"], low_memory=False)
df1 = format_reform(df1)
df1["egr"] = (df1["所有者权益合计"] - df1.groupby("证券代码").shift(1)["所有者权益合计"]) / df1.groupby("证券代码").shift(1)[
    "所有者权益合计"].replace(0, np.nan)
df1.rename({"egr": "Factor_30"}, axis="columns", inplace=True)

df2 = df1[["证券代码", "统计截止日期", "Factor_30"]]
df2.to_csv("./factor/Factor_30.csv", index=False, encoding="utf-8-sig")
