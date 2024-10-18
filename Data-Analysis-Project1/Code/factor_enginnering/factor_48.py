import pandas as pd
from util import diff
from util import format_reform
import numpy as np
# 48. nincr
# 季度频率。收益连续增加的季度数(最多8个季度)。

df1 = pd.read_csv('./data/利润表.csv', usecols=['证券代码', '报表类型', '统计截止日期', '净利润'])
df1 = format_reform(df1)

df1 = diff(df1, col=['净利润'])
df1['增长率'] = df1[['证券代码', '净利润']].groupby('证券代码').pct_change()
df1['是否增长'] = (df1['增长率'] > 0)
df1['增长分组'] = (df1['是否增长'] != df1['是否增长'].shift(1)).cumsum()
df1['增长分组'][df1['增长率'].isna()] = np.nan
df1['nincr'] = df1[['是否增长', '增长分组']].groupby('增长分组').cumsum()
df1['nincr'][df1['nincr'] > 8] = 8
df1.rename({"nincr": "Factor_48"}, axis='columns', inplace=True)

df2 = df1[['证券代码', '统计截止日期', 'Factor_48']]
df2.to_csv('./factor/Factor_48.csv', index=False, encoding='utf-8-sig')
