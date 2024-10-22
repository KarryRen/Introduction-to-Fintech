import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',default='.')
args = parser.parse_args()

d = pd.read_excel(r"TRD_Co.xlsx").iloc[2:]
d['ind'] = d['Nnindcd'].apply(lambda s:s[1:])
d = d[['Stkcd','ind']]
ind_lst = list(d.drop_duplicates(subset=['ind'])['ind'])
ind_lst.sort()
#print(ind_lst,len(ind_lst))
for id in ind_lst:
    d[str(id)] = np.where(d['ind'] == id,1,0)
#print(d)
d.to_csv(r"{:s}/industry_dummy.csv".format(args.path),encoding='utf_8_sig',index=False)