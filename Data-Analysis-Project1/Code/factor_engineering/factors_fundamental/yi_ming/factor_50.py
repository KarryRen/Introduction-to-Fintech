# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : YiMing Jiang

import pandas as pd
from util import diff
from util import format_reform

# 50. orgcap
# 季度频率。资本化管理费用。该特征使用从 CSMAR 获得的费用数据，并根据 Eisfeldt 和 Papanikolaou(2013)[5] 的定义构建。

df1 = pd.read_csv('./data/利润表.csv', usecols=['证券代码', '报表类型', '统计截止日期', '管理费用', '销售费用'])
df1 = format_reform(df1)

df1 = diff(df1, col=['管理费用', '销售费用'])
df1['orgcap'] = df1['管理费用'] + df1['销售费用']
df1.rename({"orgcap": "Factor_50"}, axis='columns', inplace=True)

df2 = df1[['证券代码', '统计截止日期', 'Factor_50']]
df2.to_csv('./factor/Factor_50.csv', index=False, encoding='utf-8-sig')
