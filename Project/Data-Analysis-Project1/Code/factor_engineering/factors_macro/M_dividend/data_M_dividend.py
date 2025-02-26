import numpy as np
import pandas as pd

#mon_lst = []
#for y in range(1990,2023):
#    for m in range(1,13):
#        mon_lst.append(int('{:d}{:02d}'.format(y,m)))
##print(mon_lst[11:-3])
#R2 = pd.DataFrame(mon_lst[11:-3],columns=['Yearmon'])
#R3 = pd.DataFrame(range(1990,2023),columns=['Year'])
#R2['Year'] = R2['Yearmon'] // 100
#R4 = pd.merge(R3,R2,on=['Year'],how='left')

#d = pd.read_excel(r"CME_Mstock2.xlsx").iloc[2:]

#d['Yearmon'] = d['Staper'].astype(str).replace('\-', '',
#regex=True).astype(int)


#d = d.rename(columns =
#              {'Esm0207': 'market_value',
#               'Esm0217': 'PE',
#               'Esm0210':'turnover'
#               })

#dt =
#d[['Yearmon','Stocksgn','market_value','turnover']].groupby(['Yearmon']).sum().reset_index()
#dt = dt[['Yearmon','market_value','turnover']]
##print(dt)
#dt['market_value'] = dt['market_value'].replace(0,np.nan)
#dt_end = dt[dt['Yearmon'] % 100 == 12][['Yearmon','market_value']]
#dt_end['Yearmon'] = dt_end['Yearmon'] // 100
#dt_end = dt_end.rename(columns={
#    'Yearmon':'Year',
#    'market_value':'year_market_value',
#    })
#dt = pd.merge(R4,dt,on=['Yearmon'],how='left')
#dt_end.loc[len(dt_end)] = {'Year':2022,'year_market_value':81162.61}
##print(dt_end)
#dt = pd.merge(dt,dt_end,on=['Year'],how='left')


#dt['M_ntis'] = (dt['market_value'] - dt['market_value'].shift(12)) /
#dt['year_market_value']

#def avg_PE(s):
#    pe_sh,pe_sz = list(s['PE'])
#    m_sh,m_sz = list(s['market_value'])
#    return (m_sh + m_sz) / (m_sh / pe_sh + m_sz / pe_sz)
#tmp =
#pd.DataFrame(d[['Yearmon','Stocksgn','PE','market_value']].groupby(['Yearmon']).apply(avg_PE).reset_index())
#tmp.columns = ['Yearmon','avg_PE']
#print(tmp)
#dt = pd.merge(dt,tmp,on=['Yearmon'],how='left')
#dt['M_ep'] = 1 / dt['avg_PE']
#dt['M_mtr'] = dt['turnover'] / dt['market_value']
##print(dt)
import os
#import akshare as ak
#all_stock = ak.stock_a_lg_indicator(symbol='all')
#all_stock.to_csv("stk.csv",encoding='utf_8_sig',index = False)
#total = len(list(all_stock['code']))
#d = pd.DataFrame()
#for i,stkcd in enumerate(list(all_stock['code'])[1000:]):
#    tmp =
#    ak.stock_a_lg_indicator(symbol=str(stkcd))[['trade_date','dv_ratio','dv_ttm','total_mv']]
#    tmp['Stkcd'] = str(stkcd)
#    d = pd.concat([d,tmp])
#    if i % 10 == 0:
#        print('----finish {:d}/{:d}----'.format(i + 1000,total))
#    if i % 1000 == 0:
#        d.to_csv("dividend__{:02d}.csv".format(1 + i //
#        1000),encoding='utf_8_sig',index = False)
#        d = pd.DataFrame()

    
#d.to_csv("dividend.csv",encoding='utf_8_sig',index = False)

# d = pd.concat([pd.read_csv('dividend' + file) for file in os.listdir('dividend')])
import os
import pandas as pd

# 获取当前工作目录
current_dir = os.getcwd()
print("当前工作目录:", current_dir)

# 构建 dividend 文件夹的路径
dividend_dir = os.path.join(current_dir, 'dividend1')

# 检查路径是否存在
if os.path.exists(dividend_dir):
    # 使用列表推导式读取所有 CSV 文件并合并
    d = pd.concat([pd.read_csv(os.path.join(dividend_dir, file)) for file in os.listdir(dividend_dir) if file.endswith('.csv')])
else:
    print(f"目录不存在: {dividend_dir}")
mon_lst = []
for y in range(1990,2025):
    for m in range(1,13):
        mon_lst.append(int('{:d}{:02d}'.format(y,m)))
#print(mon_lst[11:-3])
R2 = pd.DataFrame(mon_lst[11:-3],columns=['Yearmon'])
d['Yearmon'] = d['trade_date'].astype(str).replace('\-', '',regex=True).astype(int) // 100
d['dv_ratio'] = d['dv_ratio'].fillna(0)
d['dv_ttm'] = d['dv_ttm'].fillna(0)
d['dv'] = d[['dv_ratio','dv_ttm']].max(axis=1)
d['dv'] = d['dv'] * d['total_mv'] / 100
d = d[['Stkcd','Yearmon','dv','total_mv']]
d1 = pd.DataFrame(d.groupby(['Stkcd','Yearmon']).mean().reset_index())
d1 = pd.merge(R2,d1,on=['Yearmon'],how='left')
print(d1)
d2 = pd.DataFrame(d1.groupby(['Yearmon']).sum().reset_index())
d2['M_dp'] = d2['dv'] / d2['total_mv']
print(d2)
ep = pd.read_csv('/Users/lanyang/Desktop/Machine-Learning-in-the-Chinese-Stock-Market-Reproduction-main/lanyang/code/M_ashare/M_ep.csv')
d3 = pd.concat([pd.DataFrame([[int(ym.split('.')[0]), ep[ym][0]]],columns=['Yearmon','ep']) for ym in ep.columns])
print(d3)
d2 = pd.merge(d2,d3,on=['Yearmon'],how='left')
d2['M_de'] = d2['M_dp'] / d2['ep']
d2['M_dp'] = np.log(d2['M_dp'])
d2['M_de'] = np.log(d2['M_de'])
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

factors = ['M_dp','M_de']
for f in factors:
    transpose(d2[['Yearmon',f]]).to_csv("{:s}/{:s}.csv".format(args.path,f),encoding='utf_8_sig',index= False)