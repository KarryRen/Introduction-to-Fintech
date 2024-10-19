import pandas as pd
from util import diff
from util import if_end_of_quarter
from util import format_reform
import numpy as np
# 11. cfp
# 季度频率。营运现金流除以季度末市值。

# 12. cfp_ia
# 季度频率。经行业调整后的营运现金流。调整方法与bm_ia类似。

df1 = pd.read_csv('./data/现金流量表(直接法).csv', usecols=['证券代码', '报表类型', '统计截止日期', '经营活动产生的现金流量净额'], low_memory=False)
df1 = format_reform(df1)

df2 = pd.read_csv('./data/相对价值指标.csv', usecols=['股票代码', '统计截止日期', '市值A'])
df2 = df2[df2["统计截止日期"].apply(lambda x: if_end_of_quarter(x))]
df2.rename({'股票代码': '证券代码'}, axis='columns', inplace=True)
df2["证券代码"] = df2["证券代码"].astype(int)

df3 = pd.merge(df1, df2, on=['证券代码', '统计截止日期'], how='left')

df3 = diff(df3, col=['经营活动产生的现金流量净额'])
df3['cfp'] = df3['经营活动产生的现金流量净额']/df3['市值A'].replace(0, np.nan)

df4 = pd.read_csv('./data/公司文件.csv', usecols=['证券代码', '行业代码C'])
df4["证券代码"] = df4["证券代码"].astype(int)

df5 = pd.merge(df3, df4, on=['证券代码'], how='left')

df6 = df5[['行业代码C', '统计截止日期', 'cfp']].groupby(['行业代码C', '统计截止日期']).mean().reset_index()
df6.rename(columns={'cfp': 'cfp_i'}, inplace=True)

df7 = pd.merge(df5, df6, on=['行业代码C', '统计截止日期'], how='left')
df7['cfp_ia'] = df7['cfp'] - df7['cfp_i']
df7.rename({'cfp': 'Factor_11'}, axis='columns', inplace=True)
df7.rename({'cfp_ia': 'Factor_12'}, axis='columns', inplace=True)

df_cfp = df7[['证券代码', '统计截止日期', 'Factor_11']]
df_cfp.to_csv('./factor/Factor_11.csv', index=False, encoding='utf-8-sig')

df_cfp_ia = df7[['证券代码', '统计截止日期', 'Factor_12']]
df_cfp_ia.to_csv('./factor/Factor_12.csv', index=False, encoding='utf-8-sig')