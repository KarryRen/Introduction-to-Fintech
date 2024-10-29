import numpy as np
import pandas as pd

mon_lst = []
for y in range(1990,2025):
    for m in range(1,13):
        mon_lst.append('{:d}{:02d}'.format(y,m))
#print(mon_lst[:-4])


N = pd.read_csv(r"027nshra.csv")
N['Stkcd'] = pd.to_numeric(N['Stkcd'], errors='coerce').fillna(0).astype(int)
N['Stkcd'] = N['Stkcd'].astype(int)
N['Yearmon'] = N['Yearmon'].astype(int)
N = N[['Stkcd','Yearmon','Nshra']]

stk_lst = N.drop_duplicates(subset ='Stkcd')['Stkcd']
R1 = pd.DataFrame(stk_lst,columns=['Stkcd'])
R2 = pd.DataFrame(mon_lst[11:-4],columns=['Yearmon'])
R3 = pd.DataFrame(range(1990,2025),columns=['Year'])
R1['tmp'] = 0
R2['tmp'] = 0
R3['tmp'] = 0
R2['Yearmon'] = R2['Yearmon'].astype(int)
R4 = pd.merge(R1,R2,on=['tmp'],how='left')
del R4['tmp']
R5 = pd.merge(R1,R3,on=['tmp'],how='left')
del R5['tmp']
R6 = R2
R6['Year'] = R6['Yearmon'].astype(int) // 100
R6['Year1']=R6['Year']-1
del R6['tmp']
#print(R6)

##N = pd.read_excel(r".\CG_Capchg.xlsx")
##N = N.iloc[2:]
##N['date'] = N['Reptdt'].astype(str).replace('\-', '', regex=True)
##N['Year1'] = N['date'].astype(int) // 10000
##N = N[['Stkcd','Year1','Nshra']]
##stk_lst = N.drop_duplicates(subset ='Stkcd')['Stkcd']
##print(stk_lst)
##for s in stk_lst:
##    pre = np.nan
##    for id,row in N[N['Stkcd'] == int(s)].iterrows():
##        if np.isnan(row['Nshra']):
##            N.loc[id,'Nshra'] = pre
##        else:
##            pre = row['Nshra']
##print(N.iloc[:10])
##N=pd.merge(R6,N,on='Year1',how='left')
##N.to_csv(r"./027nshra.csv",encoding='utf_8_sig',index = False)
##print(N[N['Stkcd'] == 3])

path_027_lst = ["LT_Dailyinfo/LT_Dailyinfo_20030801_20040830.xlsx",
              "LT_Dailyinfo/LT_Dailyinfo_20040831_20090830.xlsx",
              "LT_Dailyinfo/LT_Dailyinfo_20090831_20140830.xlsx",
              "LT_Dailyinfo/LT_Dailyinfo_20140831_20190830.xlsx",
              "LT_Dailyinfo/LT_Dailyinfo_20190831_20240830.xlsx"
              ]
#path_027_lst =[r'C:\Users\lanyang\Desktop\日交易统计文件new\LT_Dailyinfo.xlsx']
d = pd.DataFrame()
for path_027 in path_027_lst: 
    print('-----reading-----')
    d = pd.concat([d,pd.read_excel(path_027).iloc[2:]])
    print('-----------------------finish {:s}'.format(path_027))
    print('-----succcess-----')

##d.to_csv(r"E:\stock\01try027.csv",encoding='utf_8_sig',index = False)
##d = pd.read_csv(r"E:\stock\try027.csv")
d = d.rename(columns =
              {'Trddt': 'Date',
               'Prccls': 'Price',##日收盘价
               'Tolstknum': 'Volume',##交易日总成交量
               'Tolstknva': 'Amount'})##交易日总成交金额
d = d[['Stkcd','Date','Price','Volume','Amount']]

d['Date'] = pd.to_datetime(d['Date'],format='%Y-%m-%d')
d['Date'] = d['Date'].astype(str).replace('\-', '', regex=True)
d['Yearmon'] = d['Date'].astype(int) // 100
d['Stkcd'] = d['Stkcd'].astype(int)
print('---------------------finish d--------------------')
#print(d.iloc[:10])
tmp = R4
print(tmp.iloc[:20])
print(tmp.dtypes)
#print(d.dtypes)
#print(N.dtypes)

# dolvol: log(Amount(t-2))
AM = d[['Stkcd','Yearmon', 'Amount']].groupby(['Stkcd','Yearmon']).sum().reset_index()
AM['dolvol'] = np.log(AM['Amount'].shift(periods=2))

tmp = pd.merge(tmp,AM[['Stkcd','Yearmon', 'dolvol']],
             on = ['Stkcd','Yearmon'],
             how='left')
print('---------------------finish dolvol--------------------')

## std_turn
STD_AM = d[['Stkcd','Yearmon','Amount']].groupby(['Stkcd','Yearmon']).std().reset_index()
STD_AM = STD_AM.rename(columns = {'Amount': 'std_turn'})
tmp = pd.merge(tmp,STD_AM[['Stkcd','Yearmon', 'std_turn']],
             on = ['Stkcd','Yearmon'],
             how='left')
print('---------------------finish std_turn--------------------')

# std_dolvol
STD_VL = d[['Stkcd','Yearmon','Volume']].groupby(['Stkcd','Yearmon']).std().reset_index()
STD_VL = STD_VL.rename(columns = {'Volume': 'std_dolvol'})
tmp = pd.merge(tmp,STD_VL[['Stkcd','Yearmon', 'std_dolvol']],
             on = ['Stkcd','Yearmon'],
             how='left')
print('---------------------finish std_dolvol--------------------')

# # turn：(Volume(t-2) + Volume(t-1) + Volume(t) ) / (3*Nshra(t))
# VL = d[['Stkcd','Yearmon','Volume']].groupby(['Stkcd','Yearmon']).sum().reset_index()
# VL['VL1'] = VL['Volume'].shift(periods=0)
# VL['VL2'] = VL['Volume'].shift(periods=1)
# VL['VL3'] = VL['Volume'].shift(periods=2)
# VL = pd.merge(VL[['Stkcd','Yearmon','VL1','VL2','VL3']],N,
#               on = ['Stkcd','Yearmon'],
#               how='left')
# VL['turn'] = (VL['VL1'] + VL['VL2'] + VL['VL3']) / (VL['Nshra'] * 3)
# tmp = pd.merge(tmp,VL[['Stkcd', 'Yearmon', 'turn']],
#                 on = ['Stkcd','Yearmon'],
#                 how='left')
# print('---------------------finish turn--------------------')

# zerotrade：count day with Volume = 0 in month t-1
M = pd.read_excel(r"TRD_Mnth.xlsx").iloc[2:]
M['date'] = M['Trdmnt'].astype(str).replace('\-', '', regex=True)
M['Yearmon'] = M['date'].astype(int)
M['Stkcd'] = M['Stkcd'].astype(int)
M = M[['Stkcd','Yearmon','Ndaytrd']]
tmp = pd.merge(tmp,M,
               on = ['Stkcd','Yearmon'],
               how='left')
ZE = d[['Stkcd','Yearmon', 'Volume']]
# ZE['zero'] = np.where(ZE['Volume'] == 0,1,0)
ZE.loc[ZE['Volume'] == 0, 'zero'] = 1
ZE.loc[ZE['Volume'] != 0, 'zero'] = 0

ZERO = ZE[['Stkcd','Yearmon',
'zero']].groupby(['Stkcd','Yearmon']).count().reset_index()
tmp = pd.merge(tmp,ZERO[['Stkcd', 'Yearmon', 'zero']],
               on = ['Stkcd','Yearmon'],
               how='left')
tmp['zerotrade'] = tmp['Ndaytrd'] - tmp['zero']
tmp['zerotrade'].shift()
print('---------------------finish zerotrade--------------------')
tmp = tmp[['Stkcd','Yearmon','dolvol','zerotrade','std_dolvol','std_turn']]
# tmp = tmp[['Stkcd','Yearmon','turn']]

import sys
sys.path.append('/Users/lanyang/Desktop/Machine-Learning-in-the-Chinese-Stock-Market-Reproduction-main/lanyang/utils')
from format_transfer import mon_freq_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',default='.')
args = parser.parse_args()

mon_freq_data(tmp,tmp.columns[2:],args.path)