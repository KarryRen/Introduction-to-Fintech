'''
Author: your name
Date: 2024-10-29 18:52:29
LastEditTime: 2024-10-29 19:17:46
LastEditors: zhangwj
Description: 
FilePath: \Introduction-to-Fintech-DAPs\Data-Analysis-Project1\Code\factor_engineering\merge_features.py
'''
import pandas as pd
import os
import numpy as np
DATA_BASE = r'Data\Factor'
# load zz500 constituent stocks
zz500 = pd.read_csv(r"zz500_stocks_240930.csv")
zz500_code = zz500['Code'].str.lower()
zz500_code = zz500_code.to_list()
# factors from daily open、 high、 low..
ohlc= pd.read_csv(os.path.join(DATA_BASE,"trading_factors.csv"))
ohlc['Date'] = ohlc['Date'].astype(str)
result=ohlc.copy()
# function to load factors from 5m open、 high、 low..
def process_min_factor(factor_file_name = ""):
    df = pd.read_hdf(os.path.join(DATA_BASE,f'{factor_file_name}.h5'),key = 'df')
    df.reset_index(inplace=True)
    df.columns = ['Code',"Date"]+ df.columns[2:].to_list()
    df['Code'] = df['Code'].str.lower()
    df = df.loc[df['Code'].isin(zz500_code)]
    df['Date'] = df['Date'].astype(str).apply(lambda x:x.replace("-",""))
    df = df.loc[(df['Date']>="20190102")&(df['Date']<="20240531")]
    return df
# factors from 5m open、 high、 low..
for min_factor in ['all_amount_factor','all_close_factor','all_turnover_factor','all_volume_factor']:
    df = process_min_factor(min_factor)
    result = pd.merge(result,df,how='left',on=['Code','Date'])
# save as float32
result[result.columns[2:]] = result[result.columns[2:]].astype('float32')
# result.to_pickle(f"{DATA_BASE}/ohlc_min.pkl")

# merge macro and fundamental factors.
jym_factor = pd.read_csv(os.path.join(DATA_BASE,"Factor.csv"),engine='pyarrow')
jym_factor.columns = ["Code","Date"]+ jym_factor.columns.to_list()[2:]
jym_factor['Date'] = jym_factor['Date'].astype('str')
jym_factor[jym_factor.columns.to_list()[2:]] = jym_factor[jym_factor.columns.to_list()[2:]].astype('float32')
jym_factor = jym_factor.groupby(by=['Code']).apply(lambda x:x.sort_values("Date").ffill(),include_groups=False)
jym_factor = jym_factor.reset_index(0)
result = pd.merge(result,jym_factor,how='left',on=['Code','Date'])
# result.to_pickle(f"{DATA_BASE}/ohlc_min_jym.pkl")

ly1 = pd.read_csv(os.path.join(DATA_BASE,"宏观11_ly.csv"),engine='pyarrow')
ly1.columns = ["Code","Date"]+ ly1.columns.to_list()[2:]
ly1['Date'] = ly1['Date'].astype('str')

ly1 = ly1.groupby(by=['Code']).apply(lambda x:x.sort_values("Date").ffill(),include_groups=False)
ly1 = ly1.reset_index(0)
ly1
result = pd.merge(result,ly1,how='left',on=['Code','Date'])
# result.to_pickle(f"{DATA_BASE}/ohlc_min_jym_lymacro.pkl")

ly2 = pd.read_pickle(os.path.join(DATA_BASE,"基本面45_ly.pkl"))
ly2_500 = ly2.loc[ly2['Stkcd'].isin(zz500_code)]
ly2_500.columns = ["Date","Code"]+ ly2_500.columns.to_list()[2:]
ly2_500['Date'] = ly2_500['Date'].astype('str')
ly2_500.replace({pd.NA: np.nan}, inplace=True)
ly2_500[ly2_500.columns.to_list()[2:]] = ly2_500[ly2_500.columns.to_list()[2:]].astype('float32')
ly2_500 = ly2_500.groupby(by=['Code']).apply(lambda x:x.sort_values("Date").ffill(),include_groups=False)
ly2_500 = ly2_500.reset_index(0)

result = pd.merge(result,ly2_500,how='left',on=['Code','Date'])
# result.to_pickle(f"{DATA_BASE}/ohlc_min_jym_lymacro_ly.pkl")
