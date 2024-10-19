import numpy as np
import pandas as pd

MV = pd.read_csv('./data/相对价值指标.csv')
MV = MV.iloc[2:]
MV = MV.rename(columns={'市值A': 'MV'})

FIT = pd.read_csv('./data/相对价值指标.csv')
FIT = FIT.iloc[2:]
FIT['Ind'] = FIT['行业代码'].apply(lambda x: str(x)[0])


def My_YM(Set, Col):
    Set['DATE'] = Set[Col].astype(str).replace('\-', '', regex=True)
    Set['Yearmon'] = Set['DATE'].astype(int) // 100


My_YM(MV, '统计截止日期')

MV['mve'] = np.log(MV['MV'].astype('float')).shift(1)
MV = pd.merge(MV, FIT[['股票代码', '统计截止日期', 'Ind']],on=['股票代码', '统计截止日期'], how='left')

MV2 = MV[['mve', 'Ind', '统计截止日期']].groupby(['Ind', '统计截止日期']).mean().reset_index()
MV2 = MV2.rename(columns={'mve': 'mveI'})

MV3 = pd.merge(MV, MV2, on=['Ind', '统计截止日期'], how='left')
MV3['mve_ia'] = MV3['mve'] - MV3['mveI']

MV3.rename({"mve": "Factor_46"}, axis='columns', inplace=True)
MV3.rename({"mve_ia": "Factor_47"}, axis='columns', inplace=True)
MV3.rename({"股票代码": "证券代码"}, axis='columns', inplace=True)

df_mve = MV3[['证券代码', '统计截止日期', 'Factor_46']]
df_mve = df_mve.dropna()
df_mve.to_csv("./factor/Factor_46.csv", encoding='utf_8_sig', index=False)

df_mve_ia = MV3[['证券代码', '统计截止日期', 'Factor_47']]
df_mve_ia = df_mve_ia.dropna()
df_mve_ia.to_csv("./factor/Factor_47.csv", encoding='utf_8_sig', index=False)
