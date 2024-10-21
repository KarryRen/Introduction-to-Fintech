import numpy as np
import pandas as pd

mon_lst = []
for y in range(1990,2025):
    for m in range(1,13):
        mon_lst.append(int('{:d}{:02d}'.format(y,m)))
#print(mon_lst[11:-3])
R2 = pd.DataFrame(mon_lst[11:-3],columns=['Yearmon'])
R3 = pd.DataFrame(range(1990,2025),columns=['Year'])
R2['Year'] = R2['Yearmon'] // 100
R4 = pd.merge(R3,R2,on=['Year'],how='left')

d = pd.read_excel(r"CME_Mstock2_new.xlsx").iloc[2:]

d['Yearmon'] = d['Staper'].astype(str).replace('\-', '', regex=True).astype(int)


d = d.rename(columns =
              {'Esm0207': 'market_value',
               'Esm0217': 'PE',
               'Esm0210':'turnover'
               })

dt = d[['Yearmon','Stocksgn','market_value','turnover']].groupby(['Yearmon']).sum().reset_index()
dt = dt[['Yearmon','market_value','turnover']]
#print(dt)
dt['market_value'] = dt['market_value'].replace(0,np.nan)
dt_end = dt[dt['Yearmon'] % 100 == 12][['Yearmon','market_value']]
dt_end['Yearmon'] = dt_end['Yearmon'] // 100
dt_end = dt_end.rename(columns={
    'Yearmon':'Year',
    'market_value':'year_market_value',
    })
dt = pd.merge(R4,dt,on=['Yearmon'],how='left')
dt_end.loc[len(dt_end)] = {'Year':2022,'year_market_value':81162.61}
#print(dt_end)
dt = pd.merge(dt,dt_end,on=['Year'],how='left')


dt['M_ntis'] = (dt['market_value'] - dt['market_value'].shift(12)) / dt['year_market_value']

def avg_PE(s):
    pe_sh,pe_sz = list(s['PE'])
    m_sh,m_sz = list(s['market_value'])
    return (m_sh + m_sz) / (m_sh / pe_sh + m_sz / pe_sz)
tmp = pd.DataFrame(d[['Yearmon','Stocksgn','PE','market_value']].groupby(['Yearmon']).apply(avg_PE).reset_index())
tmp.columns = ['Yearmon','avg_PE']
print(tmp)
dt = pd.merge(dt,tmp,on=['Yearmon'],how='left')
dt['M_ep'] = 1 / dt['avg_PE']
dt['M_mtr'] = dt['turnover'] / dt['market_value']
#print(dt)


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

factors = ['M_ntis','M_ep','M_mtr']
for f in factors:
    transpose(dt[['Yearmon',f]]).to_csv("{:s}/{:s}.csv".format(args.path,f),encoding='utf_8_sig',index = False)