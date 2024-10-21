import numpy as np
import pandas as pd

mon_lst = []
for y in range(1990,2025):
    for m in range(1,13):
        mon_lst.append(int('{:d}{:02d}'.format(y,m)))

season_lst = []
for y in range(1990,2025):
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

# 77.  sgr
# 季度频率。销售的季度百分比变化。
d2['sgr'] = (d2['sale'] - d2['sale'].shift()) / d2['sale'].shift()
d2.loc[d2['Season'] == 199012,'sgr'] = np.nan
##————————————————————
# 54.  pchgm_pchsale
# 季度频率。毛利率变动百分比减去销售变动百分比。
d2['pchgm_pchsale'] = (d2['maoli'] - d2['maoli'].shift()) / d2['maoli'].shift() - d2['sgr']
d2.loc[d2['Season'] == 199012,'pchgm_pchsale'] = np.nan

# 56.  pchsale_pchinvt
# 季度频率。销售的季度变动百分比减去存货的季度变动百分比。
d2['pchsale_pchinvt'] = d2['sgr'] - (d2['stock'] - d2['stock'].shift()) / d2['stock'].shift()
d2.loc[d2['Season'] == 199012,'pchsale_pchinvt'] = np.nan

# 57.  pchsale_pchrect
# 季度频率。销售额季度变动百分比减去应收账款季度变动百分比。
d2['pchsale_pchrect'] = d2['sgr'] - (d2['zhangkuan'] -
d2['zhangkuan'].shift()) / d2['zhangkuan'].shift()
d2.loc[d2['Season'] == 199012,'pchsale_pchrect'] = np.nan

# 58.  pchsale_pchxsga
# 季度频率。销售的季度变动百分比减去管理费用的季度变动百分比。
d2['pchsale_pchxsga'] = d2['sgr'] - (d2['manage'] - d2['manage'].shift()) / d2['manage'].shift()
d2.loc[d2['Season'] == 199012,'pchsale_pchxsga'] = np.nan

# 74.  salecash
# 季度频率。季度销售额除以现金和现金等价物。
d2['salecash'] = d2['sale'] / d2['cash']

# 75.  saleinv
# 季度频率。季度销售额除以总库存。
d2['saleinv'] = d2['sale'] / d2['stock']

# 59.  pchsaleinv
# 季度频率。销售库存比的季度百分比变化。
d2['pchsaleinv'] = (d2['saleinv'] - d2['saleinv'].shift()) / d2['saleinv'].shift()
d2.loc[d2['Season'] == 199012,'pchsaleinv'] = np.nan

# 76.  salerev
# 季度频率。季度销售额除以应收账款。
d2['salerev'] = d2['sale'] / d2['zhangkuan']
##——————————————————————-
d3 = pd.read_excel(r"FS_sale/FI_T10.xlsx").iloc[2:]
d3['date'] = d3['Accper'].astype(str).replace('\-', '', regex=True)
d3['Season'] = d3['date'].astype(int) // 100
d3 = d3[(d3['Season'] % 100) % 3 == 0]

d3['Stkcd'] = d3['Stkcd'].astype(int)
d3['Season'] = d3['Season'].astype(int)
d3 = d3.rename(columns={'F100801A':'market_value'
                    })
d3['market_value'] = d3['market_value'].replace(0,np.nan)
d3 = d3[['Stkcd','Season','market_value']]
d2 = pd.merge(d2,d3,on=['Stkcd','Season'],how='left')
##------
# 73.  rsup
# 季度频率。第t季度销售额减去第t-1季度销售额，再除以季度末市值。
d2['rsup'] = (d2['sale'] - d2['sale'].shift()) / d2['market_value']
d2.loc[d2['Season'] == 199012,'rsup'] = np.nan

# 78.  sp
# 季度频率。季度销售额除以季度末市值。
d2['sp'] = d2['sale'] / d2['market_value']

# 21.  chtx
# 季度频率。从第t-1季度到第t季度，税收的百分比变化。
d2['chtx'] = (d2['tax'] - d2['tax'].shift()) / d2['tax'].shift()
d2.loc[d2['Season'] == 199012,'chtx'] = np.nan

# 23.  currat
# 季度频率。流动资产与流动负债的比率。
##-----
d2['currat'] = d2['liudongzichan'] / d2['liudongfuzhai']
##——————————————————————————————————————
# 63.  quick
# 季度频率。速动比率=(流动资产-存货)/流动负债
d2['quick'] = (d2['liudongzichan'] - d2['stock']) / d2['liudongfuzhai']

# 55.  pchquick
# 季度频率。速动比率变动百分比。
d2['quick'] = d2['quick'].replace(0,np.nan)
d2['pchquick'] = (d2['quick'] - d2['quick'].shift()) / d2['quick'].shift()
d2.loc[d2['Season'] == 199012,'pchquick'] = np.nan

# 22.  cinvest
# 季度频率。固定资产的变化除以销售，然后取前三个季度该变量的平均值
d2['_cinvest'] = (d2['gudingzichan'] - d2['gudingzichan'].shift()) / d2['sale']
d2.loc[d2['Season'] == 199012,'_cinvest'] = np.nan
d2['cinvest'] = d2.groupby(['Stkcd'])['_cinvest'].rolling(window=3).mean().shift().tolist()
d2.loc[d2['Season'] == 199012,'cinvest'] = np.nan

# net income before extraordinary items 特别项目前收入
# extraordinary items可以认为是营业外净收入
# 净利润 + 营业外支出 - 营业外收入
# 或：营业利润 - 所得税
d2['nibei'] = d2['op_profit'] - d2['income_tax']

# 69.  roaq
# 季度频率。特别项目前收入除以滞后一季度的总资产。
d2['roaq'] = d2['nibei'] / d2['asset'].shift()
d2.loc[d2['Season'] == 199012,'roaq'] = np.nan

# 71.  roeq
# 季度频率。特别项目前收入除以滞后普通股股东权益。
# 股东权益=总资产-总负债
d2['equity'] = d2['asset'] - d2['liabilities']
d2['equity'] = d2['equity'].replace(0,np.nan)
d2['roeq'] = d2['nibei'] / d2['equity'].shift()
d2.loc[d2['Season'] == 199012,'roeq'] = np.nan

# 49.  operprof
# 季度频率。季度营业利润除以滞后一期的普通股股东权益。
d2['operprof'] = d2['op_profit'] / d2['equity'].shift()
d2.loc[d2['Season'] == 199012,'operprof'] = np.nan

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

# 24.  depr
# 季度频率。折旧除以固定资产。
# 注：折旧数据只能找到年度频率，而固定资产为季度频率，故将季度折旧设为年度折旧除以4

# 53.  pchdepr
# 半年频率。折旧变动百分比。
# 注：改为年度频率
print("r6",R6.shape)

depr = pd.read_excel(r'FS_sale/FN_Fn020.xlsx').iloc[2:]
depr['Stkcd'] = depr['Stkcd'].astype(int)
depr['date'] = depr['Accper'].astype(str).replace('\-', '', regex=True)
depr['Year'] = depr['date'].astype(int) // 10000
depr['month'] = depr['date'].apply(lambda x:x[4:6])
depr = depr[(depr['Fn02002'] == '合计') & (depr['month'] == '12')]
depr = depr.rename(columns={'Fn02003':'begin',
                           'Fn02013':'end'
                   })

depr = depr[['Stkcd','Year','begin','end']]
depr = pd.merge(R7,depr,on=['Stkcd','Year'],how='left')
depr['depr'] = (depr['end'] - depr['begin']) / 4
depr['depr'] = depr['depr'].replace(0,np.nan)
depr['pchdepr'] = (depr['depr'] - depr['depr'].shift()) / depr['depr'].shift()
depr.loc[depr['Year'] == 1990,'pchdepr'] = np.nan
depr=pd.merge(R6,depr[['Stkcd','Year','depr','pchdepr']],on=['Stkcd','Year'],how='left')
depr=depr.drop_duplicates(subset=['Stkcd','Season'])
print("depr",depr.shape)
d2=pd.merge(d2,depr[['Stkcd','Season','depr','pchdepr']],on=['Stkcd','Season'],how='left')
d2['depr'] = d2['depr'] / d2['gudingzichan']
#d2[['Stkcd','Season','depr','pchdepr']].to_csv(r"depr.csv",encoding='utf_8_sig',index=
#False)
#print(depr)

# # 64.  rd
# 季度频率。当研发费用占总资产的比例增加大于5%时，指标变量为1。
rd = pd.read_excel(r'FS_sale/PT_LCRDSPENDING.xlsx').iloc[2:]
rd = rd.rename(columns={'Symbol':'Stkcd',
                           'RDSpendSum':'rdspend'
                   })
rd['Stkcd'] = rd['Stkcd'].astype(int)
rd['date'] = rd['EndDate'].astype(str).replace('\-', '', regex=True)
rd['Year'] = rd['date'].astype(int) // 10000
rd['month'] = rd['date'].apply(lambda x:x[4:6])
rd = rd[(rd['month'] == '12')]
rd = pd.merge(R6,rd[['Stkcd','Year','rdspend']],on=['Stkcd','Year'],how='left')
print("rd",rd.shape)
d2 = pd.merge(d2,rd[['Stkcd','Season','rdspend']],on=['Stkcd','Season'],how='left')
d2['rdspend'] = d2['rdspend'] / 4
d2['rd'] = d2['rdspend'] / d2['asset']
d2['rd'] = d2['rd'].replace(0,np.nan)
d2['rd'] = (d2['rd'] - d2['rd'].shift()) / d2['rd'].shift()
d2.loc[d2['Year'] == 1990,'rd'] = np.nan
d2['rd'] = np.where(d2['rd'] > 0.05,1,0)

# 65.  rd_mve
# 季度频率。研发费用除以季度末市值。
d2['rd_mve'] = d2['rdspend'] / d2['market_value']

# 66.  rd_sale
# 季度频率。研发费用除以季度销售额。
d2['rd_sale'] = d2['rdspend'] / d2['sale']
#d2[['Stkcd','Season','rdspend','rd','rd_mve','rd_sale']].to_csv(r"depr.csv",encoding='utf_8_sig',index=
#False)

# 67.  realestate
# 季度频率。投资性房地产除以固定资产。
d2['realestate'] = d2['fangdichan'] / d2['gudingzichan']

# 70.  roavol
# 季度频率。16个季度的特别项目前收入的标准差，除以平均总资产。
d2['avg_asset'] = d2['asset'].rolling(window=16).mean()
d2['avg_asset'] = d2['avg_asset'].replace(0,np.nan)
d2['std_nibei'] = d2['nibei'].rolling(window=16).std()
d2['roavol'] = d2['std_nibei'] / d2['avg_asset']
###————————————————————————————————————————
##82.  stdcf
##季度频率。16个季度的净现金流除以销售额的标准差。
d4 = pd.read_excel(r"FS_sale/FS_Comscfd.xlsx").iloc[2:]
d4['date'] = d4['Accper'].astype(str).replace('\-', '', regex=True)
d4['Season'] = d4['date'].astype(int) // 100
d4 = d4[(d4['Season'] % 100) % 3 == 0]

d4['Stkcd'] = d4['Stkcd'].astype(int)
d4['Season'] = d4['Season'].astype(int)
d4 = d4.rename(columns={'C005000000':'cash_flow'
                    })
d4['cash_flow'] = d4['cash_flow'].replace(0,np.nan)
d4 = d4[['Stkcd','Season','cash_flow']]
d2 = pd.merge(d2,d4,on=['Stkcd','Season'],how='left')
d2['cf'] = d2['cash_flow'] / d2['sale']
d2['stdcf'] = d2['cf'].rolling(window=16).std()

# 62.  ps
# 季度频率。9个虚拟变量的和，定义类似于Piotroski(2000)[6]。根据原文的定义，是以下九个指标：
# delta表示本期减去上期：

# F_ROA，若净利润大于0则为1，否则为0
d2['F_ROA'] = np.where(d2['net_profit'] > 0,1,0)
# F_dROA，若delta净利润大于0则为1，否则为0
d2['F_dROA'] = d2['net_profit'] - d2['net_profit'].shift()
d2['F_dROA'] = np.where(d2['F_dROA'] > 0,1,0)
# F_CFO，若营运现金流大于0则为1，否则为0
d2['F_CFO'] = np.where(d2['cash_flow'] > 0,1,0)
# F_ACCRUAL，若营运现金流大于净利润则为1，否则为0
d2['F_ACCRUAL'] = np.where(d2['cash_flow'] > d2['net_profit'],1,0)
# F_dMARGIN，若d毛利润比（毛利润/总收入）大于0则为1，否则为0
d2['F_dMARGIN'] = d2['maoli'] / d2['total_sale']
d2['F_dMARGIN'] = d2['F_dMARGIN'] - d2['F_dMARGIN'].shift()
d2['F_dMARGIN'] = np.where(d2['F_dMARGIN'] > 0,1,0)
# F_dTURN，若d资产周转率（总收入/上期总资产）大于0则为1，否则为0
d2['F_dTURN'] = d2['total_sale'] / d2['asset'].shift(-1)
d2['F_dTURN'] = d2['F_dTURN'] - d2['F_dTURN'].shift()
d2['F_dTURN'] = np.where(d2['F_dTURN'] > 0,1,0)
# F_dLEVER，若d杠杆率大于0则为1，否则为0
# F_dLIQUID，若d流动比率（流动资产/流动负债）大于0则为1，否则为0
d2['F_dLIQUID'] = d2['currat'] - d2['currat'].shift()
d2['F_dLIQUID'] = np.where(d2['F_dLIQUID'] > 0,1,0)
# EQ_OFFER，若该公司在投资组合形成的前一年没有发行普通股则为1，否则为0
d2['ps'] = d2['F_ROA'] + d2['F_dROA'] + d2['F_CFO'] + d2['F_ACCRUAL'] + d2['F_dMARGIN'] + d2['F_dTURN'] + d2['F_dLIQUID']


# 84.  tb
# 季度频率。税收收入，定义为当期税收费用除以中国企业所得税税率25%，除以总收入。
d2['tb'] = 4 * d2['income_tax'] / d2['total_sale']

d2 = d2[['Stkcd','Season',
         'chtx',
         'cinvest',
         'currat',
         'quick',
         'pchquick',
         'roaq',
         'roeq',
         'operprof',
         'pchgm_pchsale',
         'pchsale_pchinvt',
         'pchsale_pchrect',
         'pchsale_pchxsga',
         'pchsaleinv',
         'rsup',
         'salecash',
         'saleinv',
         'salerev',
         'sgr',
         'sp',
         'depr',
         'pchdepr',
         'rd',
         'rd_mve',
         'rd_sale',
         'realestate',
         'roavol',
         'stdcf',
         'ps',
         'tb',
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