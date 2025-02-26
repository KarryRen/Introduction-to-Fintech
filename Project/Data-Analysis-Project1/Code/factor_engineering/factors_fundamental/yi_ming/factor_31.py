# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : YiMing Jiang

import pandas as pd
from util import diff
from util import format_reform
import numpy as np

# 31. gma
# 季度频率。收入减去销货成本后，除以滞后一期的总资产

df1 = pd.read_csv("./data/资产负债表.csv", usecols=["证券代码", "报表类型", "统计截止日期", "资产总计"], low_memory=False)
df1 = format_reform(df1)

df2 = pd.read_csv("./data/现金流量表(直接法).csv",
                  usecols=["证券代码", "报表类型", "统计截止日期", "销售商品、提供劳务收到的现金", "购买商品、接受劳务支付的现金"], low_memory=False)
df2 = format_reform(df2)

df3 = pd.merge(df1, df2, on=["证券代码", "统计截止日期"], how="left")
df3["证券代码"] = df3["证券代码"].astype(int)
df3 = diff(df3, col=["销售商品、提供劳务收到的现金", "购买商品、接受劳务支付的现金"])
df3["gma"] = (df3["销售商品、提供劳务收到的现金"] - df3["购买商品、接受劳务支付的现金"]) / df3.groupby("证券代码").shift(1)["资产总计"].replace(0,
                                                                                                                                              np.nan)
df3.rename({"gma": "Factor_31"}, axis="columns", inplace=True)

df4 = df3[["证券代码", "统计截止日期", "Factor_31"]]
df4.to_csv("./factor/Factor_31.csv", index=False, encoding="utf-8-sig")
