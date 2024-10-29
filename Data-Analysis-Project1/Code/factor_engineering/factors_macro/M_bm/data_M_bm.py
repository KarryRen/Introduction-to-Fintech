import numpy as np
import pandas as pd

mon_lst = []
for y in range(1990,2025):
    for m in range(1,13):
        mon_lst.append(int('{:d}{:02d}'.format(y,m)))
#print(mon_lst[11:-3])
season_lst = []
for y in range(1990,2024):
    for m in range(3,13,3):
        season_lst.append(int('{:d}{:02d}'.format(y,m)))


d = pd.read_excel(r"FI_T10.xlsx").iloc[2:]
d['date'] = d['Accper'].astype(str).replace('\-', '', regex=True)
d['Season'] = d['date'].astype(int) // 100
d = d[(d['Season'] % 100) % 3 == 0]

d['Stkcd'] = d['Stkcd'].astype(int)
d['Season'] = d['Season'].astype(int)
d = d.rename(columns={'F100801A':'market',
                    'F101001A':'book'
                    })
d = d[['Stkcd','Season','market','book']]
d['book'] = d['book'] * d['market']
R2 = pd.DataFrame(mon_lst[11:-3],columns=['Yearmon'])
R3 = pd.DataFrame(season_lst[3:-1],columns=['Season'])
R2['num'] = R2['Yearmon'] // 100 * 10 + ((R2['Yearmon'] % 100) - 1) // 3
R3['num'] = R3['Season'] // 100 * 10 + (R3['Season'] % 100) // 3 - 1
R4 = pd.merge(R2,R3,on=['num'],how='left')

print(R3)
print(d)
tmp = pd.merge(R3,pd.DataFrame(d[['Season','market','book']].groupby(['Season']).sum().reset_index()),how='left',on=['Season'])

tmp['M_bm'] = tmp['book'] / tmp['market']
tmp=pd.merge(R4[['Yearmon','Season']],tmp[['Season','M_bm']],how='left',on=['Season'])
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

transpose(tmp[['Yearmon','M_bm']]).to_csv("{:s}/M_bm.csv".format(args.path),encoding='utf_8_sig',index = False)