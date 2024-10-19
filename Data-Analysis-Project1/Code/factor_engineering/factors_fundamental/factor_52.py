import pandas as pd
from util import format_reform
import numpy as np

df1 = pd.read_csv('./data/资产负债表.csv', usecols=['证券代码', '报表类型', '统计截止日期', '流动资产合计', '流动负债合计', '存货净额'], low_memory=False)
df1 = format_reform(df1)

# 23.  currat
# 季度频率。流动资产与流动负债的比率。
df1['currat'] = df1['流动资产合计']/df1['流动负债合计'].replace(0, np.nan)

# 52. pchcurrat
# 季度频率。流动比率（流动负债除以流动资产）的变动百分比。
df1['pchcurrat'] = (df1['currat'] - df1.groupby('证券代码').shift(1)['currat'])/df1.groupby('证券代码').shift(1)['currat'].replace(0, np.nan)

# 63.  quick
# 季度频率。速动比率=(流动资产-存货)/流动负债
df1['quick'] = (df1['流动资产合计'] - df1['存货净额'])/df1['流动负债合计'].replace(0, np.nan)

# 55.  pchquick
# 季度频率。速动比率变动百分比。
df1['quick'] = df1['quick'].replace(0, np.nan)
df1['pchquick'] = (df1['quick'] - df1.groupby('证券代码').shift(1)['quick'])/df1.groupby('证券代码').shift(1)['quick'].replace(0, np.nan)

df1.rename({"currat": "Factor_23"}, axis='columns', inplace=True)
df1.rename({"pchcurrat": "Factor_52"}, axis='columns', inplace=True)
df1.rename({"pchquick": "Factor_55"}, axis='columns', inplace=True)
df1.rename({"quick": "Factor_63"}, axis='columns', inplace=True)


df_pchcurrat = df1[['证券代码', '统计截止日期', 'Factor_52']]
df_pchcurrat.to_csv('./factor/Factor_52.csv', index=False, encoding='utf-8-sig')