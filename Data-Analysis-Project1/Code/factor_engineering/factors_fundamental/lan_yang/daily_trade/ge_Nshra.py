import numpy as np
import pandas as pd
mon_lst = []
for y in range(1990,2023):
    for m in range(1,13):
        mon_lst.append('{:d}{:02d}'.format(y,m))
#print(mon_lst[:-4])




R2 = pd.DataFrame(mon_lst[11:-4],columns=['Yearmon'])
R3 = pd.DataFrame(range(1990,2023),columns=['Year'])

R2['tmp'] = 0
R3['tmp'] = 0
R2['Yearmon'] = R2['Yearmon'].astype(int)

R6 = R2
R6['Year'] = R6['Yearmon'].astype(int) // 100
R6['Year1']=R6['Year']-1
del R6['tmp']

N = pd.read_excel(r"CG_Capchg.xlsx")
N = N.iloc[2:]
N['date'] = N['Reptdt'].astype(str).replace('\-', '', regex=True)
N['Year1'] = N['date'].astype(int) // 10000
N = N[['Stkcd','Year1','Nshra']]
stk_lst = N.drop_duplicates(subset ='Stkcd')['Stkcd']
print(stk_lst)
for s in stk_lst:
   pre = np.nan
   for id,row in N[N['Stkcd'] == int(s)].iterrows():
       if np.isnan(row['Nshra']):
           N.loc[id,'Nshra'] = pre
       else:
           pre = row['Nshra']
print(N.iloc[:10])
N=pd.merge(R6,N,on='Year1',how='left')
N.to_csv(r"./027nshra.csv",encoding='utf_8_sig',index = False)
print(N[N['Stkcd'] == 3])
