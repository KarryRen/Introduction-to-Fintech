import pandas as pd
import numpy as np
from util import format_reform
from util import diff

# 32. grCAPX
# 半年频率。资本支出从t-2到t年的百分比变化。

df1 = pd.read_csv("./data/现金流量表(直接法).csv", usecols=["证券代码", "报表类型", "统计截止日期", "购建固定资产、无形资产和其他长期资产支付的现金"],
                  low_memory=False)
df1 = format_reform(df1)
df1["半年"] = df1["统计截止日期"].apply(lambda x: True if x[5:7] in ["06", "12"] else False)

df2 = df1[df1["半年"]].reset_index().drop(columns=["index"])
df2["证券代码"] = df2["证券代码"].astype(int)
df2 = diff(df2, col=["购建固定资产、无形资产和其他长期资产支付的现金"], freq="半年")
df2["grCAPX"] = (df2["购建固定资产、无形资产和其他长期资产支付的现金"] - df2.groupby("证券代码").shift(4)[
    "购建固定资产、无形资产和其他长期资产支付的现金"]) / df2.groupby("证券代码").shift(4)["购建固定资产、无形资产和其他长期资产支付的现金"].replace(0,
                                                                                                                                                  np.nan)
df2.rename({"grCAPX": "Factor_32"}, axis="columns", inplace=True)

df3 = df2[["证券代码", "统计截止日期", "Factor_32"]]
df3.to_csv("./factor/Factor_32.csv", index=False, encoding="utf-8-sig")
