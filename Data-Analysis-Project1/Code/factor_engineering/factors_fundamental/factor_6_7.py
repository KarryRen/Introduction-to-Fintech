import pandas as pd

# 06. bm
# 季度频率。账面市值比，等于权益的账面价值除以市值。

# 07. bm_ia
# 季度频率。经行业调整的账面市值比。

df1 = pd.read_csv('./data/相对价值指标.csv', usecols=['股票代码', '统计截止日期', '账面市值比A'])
df1.rename({'账面市值比A': 'bm'}, axis='columns', inplace=True)
df1.rename({'股票代码': '证券代码'}, axis='columns', inplace=True)

df2 = pd.read_csv('./data/公司文件.csv')[['证券代码', '行业代码C']]

df3 = pd.merge(df1, df2, on=['证券代码'], how='left')

df4 = df3[['行业代码C', '统计截止日期', 'bm']].groupby(['行业代码C', '统计截止日期']).mean().reset_index()
df4.rename(columns={'bm': 'bm_i'}, inplace=True)

df5 = pd.merge(df3, df4, on=['行业代码C', '统计截止日期'], how='left')
df5['bm_ia'] = df5['bm'] - df5['bm_i']

# df6 = df5[['证券代码', '统计截止日期', 'bm', 'bm_ia']]
df5.rename({'bm': 'Factor_6'}, axis='columns', inplace=True)
df5.rename({'bm_ia': 'Factor_7'}, axis='columns', inplace=True)

df_bm = df5[['证券代码', '统计截止日期', 'Factor_6']]
df_bm.to_csv('./factor/Factor_6.csv', index=False, encoding='utf-8-sig')

df_bm_ia = df5[['证券代码', '统计截止日期', 'Factor_7']]
df_bm_ia.to_csv('./factor/Factor_7.csv', index=False, encoding='utf-8-sig')
