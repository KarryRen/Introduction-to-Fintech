import pandas as pd
import numpy as np

d = pd.read_excel(r'HLD_NegCR.xlsx').iloc[2:]
season_lst = []
for y in range(1990,2025):
    for m in range(3,13,3):
        season_lst.append(int('{:d}{:02d}'.format(y,m)))


d['date'] = d['Reptdt'].astype(str).replace('\-', '', regex=True)
d['Season'] = d['date'].astype(int) // 100
d = d[(d['Season'] % 100) % 3 == 0]
d['Stkcd'] = d['Stkcd'].astype(int)
d['Season'] = d['Season'].astype(int)

stk_lst = d.drop_duplicates(subset ='Stkcd')['Stkcd']
R1 = pd.DataFrame(stk_lst,columns=['Stkcd'])
R3 = pd.DataFrame(season_lst[3:-1],columns=['Season'])
R1['tmp'] = 0
R3['tmp'] = 0

R5 = pd.merge(R1,R3,on=['tmp'],how='left')
del R5['tmp']

d = pd.merge(R5,d,on=['Stkcd','Season'],how='left')
d = d.rename(columns={'Negshrcr1':'largestholderrate',
                    'Negshrcr4':'top10holderrate'
    })
d = d[['Stkcd','Season','largestholderrate','top10holderrate']]
d['largestholderrate'] = d['largestholderrate'] / 100
d['top10holderrate'] = d['top10holderrate'] / 100

import sys
sys.path.append('../../utils')
from format_transfer import season_freq_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',default='.')
args = parser.parse_args()

season_freq_data(d,d.columns[2:],args.path)