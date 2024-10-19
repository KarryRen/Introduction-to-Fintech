"""
受限于公司文件，有部分股票不在其中，导致部分股票没有33号因子
"""
import pandas as pd
from util import diff
from util import format_reform
import numpy as np
# 33. herf
# 季度频率。行业内各公司的销售百分比的平方和。

df1 = pd.read_csv('./data/利润表.csv', usecols=['证券代码', '报表类型', '统计截止日期', '营业总收入'])
df1 = format_reform(df1)

df2 = pd.read_csv('./data/公司文件.csv', usecols=['证券代码', '行业代码C'])
df2["证券代码"] = df2["证券代码"].astype(int)

df3 = pd.merge(df1, df2, on=['证券代码'], how='left')
df3 = diff(df3, col=['营业总收入'])

df4 = df3[['行业代码C', '统计截止日期', '营业总收入']].groupby(['行业代码C', '统计截止日期']).sum()
df4.rename({'营业总收入':'行业营业总收入'}, axis='columns', inplace=True)

df5 = pd.merge(df3, df4, on=['行业代码C', '统计截止日期'], how='left')
df5['herf'] = (df5['营业总收入']/df5['行业营业总收入'].replace(0, np.nan))**2
df5.rename({"herf": "Factor_33"}, axis='columns', inplace=True)

df6 = df5[['证券代码', '统计截止日期', 'Factor_33']]
df6.to_csv('./factor/Factor_33.csv', index=False, encoding='utf-8-sig')
