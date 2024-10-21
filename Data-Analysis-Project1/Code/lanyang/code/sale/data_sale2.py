import numpy as np
import pandas as pd

mon_lst = []
for y in range(1990,2025):
    for m in range(1,13):
        mon_lst.append(int('{:d}{:02d}'.format(y,m)))

season_lst = []
for y in range(1990,2023):
    for m in range(3,13,3):
        season_lst.append(int('{:d}{:02d}'.format(y,m)))

d = pd.read_excel(r"FS_sale/FS_Comins.xlsx").iloc[2:]

d['date'] = d['Accper'].astype(str).replace('\-', '', regex=True)
d['Season'] = d['date'].astype(int) // 100
d = d[(d['Season'] % 100) % 3 == 0]

d['Stkcd'] = d['Stkcd'].astype(int)
d['Season'] = d['Season'].astype(int)
d = d.rename(columns={'B001101000':'sale',
                    'B001100000':'total_sale',
                    'B001000000':'maoli',#利润总额（毛利）
                    'B001210000':'manage',
                    'B001300000':'op_profit',#营业利润
                    'B002100000':'income_tax'
                    })

d['sale'] = d['sale'].fillna(1e20)
d['sale'] = d[['sale','total_sale']].min(axis=1)
d['sale'] = d['sale'].replace(0,np.nan)
d['maoli'] = d['maoli'].replace(0,np.nan)
d['manage'] = d['manage'].replace(0,np.nan)
d = d[['Stkcd','Season','sale','maoli','manage','op_profit','income_tax','total_sale']]
d['net_profit'] = d['maoli'] - d['income_tax']
d['total_sale'] = d['total_sale'].replace(0,np.nan)

stk_lst = d.drop_duplicates(subset ='Stkcd')['Stkcd']
R1 = pd.DataFrame(stk_lst,columns=['Stkcd'])
R2 = pd.DataFrame(mon_lst[11:-3],columns=['Yearmon'])
R3 = pd.DataFrame(season_lst[3:-1],columns=['Season'])
RY = pd.DataFrame(range(1990,2023),columns=['Year'])
R1['tmp'] = 0
R2['tmp'] = 0
R3['tmp'] = 0
RY['tmp'] = 0
R2['Yearmon'] = R2['Yearmon'].astype(int)
R4 = pd.merge(R1,R2,on=['tmp'],how='left')
del R4['tmp']
R5 = pd.merge(R1,R3,on=['tmp'],how='left')
del R5['tmp']
R6 = R5
R5['Year'] = R5['Season'].astype(int) // 100
R7 = pd.merge(R1,RY,on=['tmp'],how='left')
del R7['tmp']

d = pd.merge(R5,d,on=['Stkcd','Season'],how='left')
print('-----finish d-----')
#print(d[:20])
d1 = pd.read_excel(r"FS_sale/FS_Combas.xlsx").iloc[2:]
d1['date'] = d1['Accper'].astype(str).replace('\-', '', regex=True)
d1['Season'] = d1['date'].astype(int) // 100
d1 = d1[(d1['Season'] % 100) % 3 == 0]
d1['Stkcd'] = d1['Stkcd'].astype(int)
d1['Season'] = d1['Season'].astype(int)
d1 = d1.rename(columns={'A001101000':'cash',
                    'A002113000':'tax',
                    'A001123000':'stock',
                    'A001111000':'zhangkuan',
                    'A001100000':'liudongzichan',
                    'A002100000':'liudongfuzhai',
                    'A001212000':'gudingzichan',
                    'A001000000':'asset',
                    'A002000000':'liabilities',
                    'A001211000':'fangdichan'
})
d1 = d1[['Stkcd','Season','cash','tax','stock','zhangkuan','liudongzichan','liudongfuzhai','gudingzichan','asset','liabilities','fangdichan']]
d1['stock'] = d1['stock'].replace(0,np.nan)
d1['cash'] = d1['cash'].replace(0,np.nan)
d1['zhangkuan'] = d1['zhangkuan'].replace(0,np.nan)
d1['tax'] = d1['tax'].replace(0,np.nan)
d1['liudongfuzhai'] = d1['liudongfuzhai'].replace(0,np.nan)
d1['asset'] = d1['asset'].replace(0,np.nan)
d1['gudingzichan'] = d1['gudingzichan'].replace(0,np.nan)
d2 = pd.merge(d,d1,on=['Stkcd','Season'],how='left')
print('-----finish d1-----')
##print(d1[:20])


d2['nibei'] = d2['op_profit'] - d2['income_tax']


# 19.  chpm
# 季度频率。特别项目前收入的变化除以总资产。
d2['chpm'] = (d2['nibei'] - d2['nibei'].shift()) / d2['asset']
d2.loc[d2['Season'] == 199012,'chpm'] = np.nan

#20.  chpm_ia
#季度频率。行业调整后的 chpm。
ind = pd.read_excel(r'FS_sale/TRD_Co.xlsx').iloc[2:]
ind['Ind'] = ind['Nnindcd'].apply(lambda x: x[0])
ind['Stkcd'] = ind['Stkcd'].astype(int)
d2 = pd.merge(d2,ind[['Stkcd','Ind']],on=['Stkcd'],how='left')
#print(d2)
# FF2 = d2[['chpm', 'Ind', 'Season']].groupby(['Ind', 'Season']).apply(lambda s:s.sum() / s.count())
# FF2 = d2[['chpm', 'Ind', 'Season']].groupby(['Ind', 'Season']).apply(
#     lambda s: s['chpm'].sum() / s['chpm'].count() if s['chpm'].count() > 0 else np.nan
# ).reset_index()
FF2 = d2[['chpm', 'Ind', 'Season']].groupby(['Ind', 'Season']).mean().reset_index()
# df6 = df5[['行业代码C', '会计期间', 'chpm']].groupby(['行业代码C', '会计期间']).mean().reset_index()
FF2 = FF2.rename(columns = {'chpm': 'chpmI'})
d2 = pd.merge(d2,FF2,on=['Ind', 'Season'],how='left')
# 检查 'chpmI' 列是否存在于 d2 中
if 'chpmI' in d2.columns:
    d2['chpm_ia'] = d2['chpm'] - d2['chpmI']
else:
    print("Column 'chpmI' does not exist in d2.")

# d2['chpm_ia'] = d2['chpm'] - d2['chpmI']


d2 = d2[['Stkcd','Season',
         'chpm_ia',
         'chpm'
         ]]
print(d2)
import sys
sys.path.append('/Users/lanyang/Desktop/Machine-Learning-in-the-Chinese-Stock-Market-Reproduction-main/lanyang/utils')
from format_transfer import season_freq_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',default = '.')
args = parser.parse_args()

season_freq_data(d2,d2.columns[2:],args.path)