import numpy as np
import pandas as pd

#import akshare as ak
#d = ak.bond_zh_us_rate()
#d.to_csv("debt.csv",encoding='utf_8_sig',index = False)
mon_lst = []
for y in range(1990,2025):
    for m in range(1,13):
        mon_lst.append(int('{:d}{:02d}'.format(y,m)))
#print(mon_lst[11:-3])
d = pd.read_excel('BND_TreasYield_new.xlsx').iloc[2:]
d['Yearmon'] = d['Trddt'].astype(str).replace('\-', '', regex=True).astype(int) // 100
d['Yeartomatu'] = d['Yeartomatu'].astype(str)
print(d['Yeartomatu'])
d1 = d[d['Yeartomatu'] == '1.0'][['Yearmon','Yield']]
print(d1)
tmp1 = pd.DataFrame(d1.groupby(['Yearmon']).apply(lambda s:s['Yield'].sum() / s['Yield'].count()),columns=['r1']).reset_index()
R2 = pd.DataFrame(mon_lst[11:-3],columns=['Yearmon'])
# print(tmp1)
# print(R2)
tmp1 = pd.merge(R2,tmp1,on=['Yearmon'],how='left')

d10 = d[d['Yeartomatu'] == '10.0'][['Yearmon','Yield']]
tmp10 = pd.DataFrame(d10.groupby(['Yearmon']).apply(lambda s:s['Yield'].sum() / s['Yield'].count()),columns=['r10']).reset_index()
tmp10 = pd.merge(R2,tmp10,on=['Yearmon'],how='left')
tmp1 = pd.merge(tmp1,tmp10,on=['Yearmon'],how='left')
print(tmp1)
tmp1['r10_minus_r1'] = tmp1['r10'] - tmp1['r1']
#d = pd.read_csv(r"debt.csv")
#d['Yearmon'] = d['日期'].astype(str).replace('\-', '', regex=True).astype(int)
#// 100
#d = d.rename(columns={'中国国债收益率10年-2年':'M_tms'
#                    })
#d=d[['Yearmon','M_tms']]
#print(d)
#R2 = pd.DataFrame(mon_lst[11:-3],columns=['Yearmon'])
#tmp = pd.DataFrame(d.groupby(['Yearmon']).mean().reset_index())
##tmp = pd.DataFrame(d.groupby(['Yearmon']).apply(lambda
##s:s['M_tms'].sum()/s['M_tms'].count()).reset_index())
#tmp = pd.merge(R2,tmp,on=['Yearmon'],how='left')
#print(tmp)
def transpose(tmp):
    ret = pd.DataFrame()
    factor = tmp.columns[1]
    for id,row in tmp.iterrows():
        ret[row['Yearmon']] = [row[factor]]
    return ret

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',default='.')
args = parser.parse_args()

transpose(tmp1[['Yearmon','r1']]).to_csv("{:s}/r1.csv".format(args.path),encoding='utf_8_sig',index = False)
transpose(tmp1[['Yearmon','r10_minus_r1']]).to_csv("{:s}/M_tms.csv".format(args.path),encoding='utf_8_sig',index = False)
