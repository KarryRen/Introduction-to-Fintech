import numpy as np
import pandas as pd

mon_lst = []
for y in range(1990,2025):
    for m in range(1,13):
        mon_lst.append(int('{:d}{:02d}'.format(y,m)))
#print(mon_lst[11:-3])
R2 = pd.DataFrame(mon_lst[11:],columns=['Yearmon'])

d = pd.read_excel(r"TRD_Index.xlsx").iloc[2:]

d['Yearmon'] = d['Trddt'].astype(str).replace('\-', '', regex=True).astype(int) // 100
d['r2'] = d['Retindex'] ** 2
print(d)
tmp = pd.DataFrame(d[['Yearmon','r2']].groupby(['Yearmon']).apply(lambda s:s['r2'].sum()/s['r2'].count()).reset_index())
tmp = pd.merge(R2,tmp,on=['Yearmon'],how='left')

tmp.columns = ['Yearmon','M_svar']
print(tmp)



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

transpose(tmp).to_csv("{:s}/M_svar.csv".format(args.path),encoding='utf_8_sig',index = False)