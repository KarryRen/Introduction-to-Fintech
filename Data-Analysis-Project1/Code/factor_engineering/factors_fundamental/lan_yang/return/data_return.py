import pandas as pd
import numpy as np
import os

# path_040_lst = ["TRD_Dalyr/TRD_Dalyr.xlsx"]
#for files in os.listdir('TRD_Dalyr'):
#    for f in os.listdir('TRD_Dalyr/' + files):
#        if f.split('.')[-1] == 'xlsx':
#            path_040_lst.append('/'.join(['TRD_Dalyr',files,f]))
#print(len(path_040_lst))

path_040_lst = []
for f in os.listdir('TRD_Dalyr'):
#    for f in os.listdir('TRD_Dalyr/' + files):
       if f.split('.')[-1] == 'xlsx':
           path_040_lst.append('/'.join(['TRD_Dalyr',f]))
print(path_040_lst)

d = pd.DataFrame()
for path_040 in path_040_lst: 
    print('-----reading-----')
    d = pd.concat([d,pd.read_excel(path_040)[['Stkcd', 'Trddt', 'Dretnd']].iloc[2:]])##不考虑现金红利的日个股回报率
    print('-----------------------finish {:s}'.format(path_040))
    print('-----success-----')
d = d.rename(columns = {'Dretnd': 'ret'})



# df1 = pd.read_csv('data/日个股回报率/daily_return.csv')[["Stkcd","Trddt","Dsmvosd","Dsmvtll","Dretwd","Clsprc"]]
#     # df1 = df1[-1000:].reset_index()
#     df1.rename({'Stkcd':"证券代码",'Trddt':"会计期间","Dsmvosd":"流通市值","Dsmvtll":"总市值","Dretwd":"回报率","Clsprc":"收盘价"},axis='columns',inplace=True)
#     df1["year"] = pd.DatetimeIndex(df1["会计期间"]).year.astype(str)
#     # df1["year"] = df1["year"].astype(int)
#     df1["year"] = df1["year"].astype(str)
#     df1.reset_index(drop=True,inplace=True)
#     df1["month"] = pd.DatetimeIndex(df1["会计期间"]).month.astype(str)
#     for i in range(len(df1)):
#         if len(df1.loc[i,"month"]) == 1:
#              df1.loc[i,"month"] = '0'+df1.loc[i,"month"]

#     #40 最大日回报
#     df2 = df1
#     df2["会计期间"] = df2["year"]+df2["month"]
#     df2 = df2[["证券代码","会计期间","回报率"]].groupby(["证券代码","会计期间"]).max().reset_index()
#     # df2["回报率"] = df2["回报率"].shift(1)
#     df2.loc[0,"回报率"] = None
#     for i in range(1,len(df2)):
#         if df2.loc[i,"证券代码"] != df2.loc[i-1,"证券代码"]:
#             df2.loc[i,"回报率"] = None
#     df2.rename({"回报率":"maxret"},axis='columns',inplace=True)
#     df2["会计期间"] = [x.date() for x in pd.to_datetime(df2["会计期间"],format='%Y%m')]
#     df2.to_csv('data/月_40.csv',index=False,encoding='utf-8_sig')

d['Yearmon'] = d['Trddt'].astype('str').replace('\-', '', regex=True)

d['Yearmon'] = d['Yearmon'].astype('int') // 100
d['Yearmon'] = d['Yearmon'].astype('str')
# d.reset_index(drop=True,inplace=True)
mon_lst = []
for y in range(2008,2024):
    for m in range(1,13):
        mon_lst.append('{:d}{:02d}'.format(y,m))
# print(mon_lst[8:12])
stk_lst = d.drop_duplicates(subset ='Stkcd')['Stkcd']
R1 = pd.DataFrame(stk_lst,columns=['Stkcd'])
# R2 = pd.DataFrame(mon_lst[8:12],columns=['Yearmon'])
# d = pd.merge(R2,d,how='left',on='Yearmon')

d['ret'] = pd.to_numeric(d['ret'], errors='coerce')
# print(d.dtypes)
maxret = d[['Stkcd','Yearmon','ret']].groupby(['Stkcd','Yearmon']).max().reset_index()
# print(maxret)
maxret = maxret.rename(columns = {'ret': 'maxret'})
sumret = d[['Stkcd','Yearmon','ret']].groupby(['Stkcd','Yearmon']).sum().reset_index()
sumret = sumret.rename(columns = {'ret': 'sumret'})
std = d[['Stkcd','Yearmon','ret']].groupby(['Stkcd','Yearmon']).std().reset_index()
std = std.rename(columns = {'ret': 'std'})


d = pd.merge(d,maxret,how='left',on=['Stkcd','Yearmon'])
d = pd.merge(d,sumret,how='left',on=['Stkcd','Yearmon'])
d = pd.merge(d,std,how='left',on=['Stkcd','Yearmon'])

d1 = d.drop_duplicates(subset=['Stkcd','Yearmon'])

# R2['tmp'] = 0
# R1['tmp'] = 0
# R = pd.merge(R1,on=['tmp'],how='left')
# #print(R)
# d1 = pd.merge(R[['Stkcd','Yearmon']],d1,how='left',on=['Stkcd','Yearmon'])
d1 = d1.sort_values(by=['Stkcd','Yearmon']).reset_index()
d1 = d1[['Stkcd','Yearmon','maxret','sumret','std']]

d1['Yearmon'] = d1['Yearmon'].astype('str')
print('-----------------------------finish d1')
# print(d1.loc[d1['Yearmon'] == '199012'])

# maxret
d1['maxret'] = d1['maxret'].shift()

# ground truth
d1['ground_truth'] = d1['sumret']

# mom1m
d1['sumret'] = d1['sumret'].shift()
d1 = d1.rename(columns={'sumret':'mom1m'})

# volatility
d1['std'] = d1['std'].shift()
d1 = d1.rename(columns={'std':'volatility'})

# 清空错位数据
d1.loc[d1['Yearmon'] == '202209',['maxret','mom1m','volatility']] = np.nan

# mom6m
m6 = d1.groupby(['Stkcd'])['mom1m'].rolling(window=5).sum().shift()
d1['mom6m'] = m6.tolist()
print('-----------------------------finish mom6m')

# mom12m
m12 = d1.groupby(['Stkcd'])['mom1m'].rolling(window=11).sum().shift()
d1['mom12m'] = m12.tolist()
print('-----------------------------finish mom12m')

# mom36m
t36 = d1.groupby(['Stkcd'])['mom1m'].rolling(window=36).sum().shift()
t12 = d1.groupby(['Stkcd'])['mom1m'].rolling(window=12).sum().shift()
m36 = t36 - t12
d1['mom36m'] = m36.tolist()
print('-----------------------------finish mom36m')

d1.loc[d1['Yearmon'] == '202209',['mom6m','mom12m','mom36m']] = np.nan
d1['Yearmon']=d1['Yearmon'].astype(int)
d1['Stkcd']=d1['Stkcd'].astype(int)
print(d1.dtypes)
import sys
sys.path.append('/Users/lanyang/Desktop/Machine-Learning-in-the-Chinese-Stock-Market-Reproduction-main/lanyang/utils')
from format_transfer import mon_freq_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',default='.')
args = parser.parse_args()

mon_freq_data(d1,d1.columns[2:],args.path)
