import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',default='.')
args = parser.parse_args()

def transpose(tmp):
    ret = pd.DataFrame()
    factor = tmp.columns[1]
    for id,row in tmp.iterrows():
        ret[row['Yearmon']] = [row[factor]]
    return ret

d = pd.read_excel(r'CME_Mpi1_new.xlsx')
d['Yearmon'] = d['Staper'].astype('str').replace('\-', '', regex=True)
#print(d.dtypes)
d = d.loc[(d['Fresgn'] == 'M') & (d['Datasgn'] == 'PYM') & (d['Areasgn'] =='1')]
d = d.rename(columns={'Epim0101':'CPI'})
d = d[['Yearmon','CPI']][d['Yearmon'] >= '199012']
print(d)
d = transpose(d)
d.to_csv(r"{:s}/M_infl.csv".format(args.path),encoding='utf_8_sig',index = False)

d = pd.read_excel(r'CME_Mfinamkt1_new.xlsx')
d['Yearmon'] = d['Staper'].astype('str').replace('\-', '', regex=True)
#print(d.dtypes)
d = d.loc[(d['Fresgn'] == 'M') & (d['Datasgn'] == 'B')]
d = d.rename(columns={'Ezm0109':'M2'})
d = d[['Yearmon','M2']][d['Yearmon'] >= '199012']
print(d)
d = transpose(d)
d.to_csv(r"{:s}/M_m2gr.csv".format(args.path),encoding='utf_8_sig',index = False)

d = pd.read_excel(r'CME_Mftrd1_new.xlsx')
d['Yearmon'] = d['Staper'].astype('str').replace('\-', '', regex=True)
print(d.dtypes)
d = d.loc[(d['Fresgn'] == 'M') & (d['Datasgn'] == 'A')]
d = d.rename(columns={'Eftm0101':'trade'})
d['itgr'] = (d['trade'] - d['trade'].shift(12)) / d['trade'].shift(12)
d = d[['Yearmon','itgr']][d['Yearmon'] >= '199012']
print(d)
d = transpose(d)
d.to_csv(r"{:s}/M_itgr.csv".format(args.path),encoding='utf_8_sig',index = False)