import pandas as pd


df1 = pd.read_excel('TRD_Dalyr.xlsx')[["Stkcd","Trddt","Dsmvosd","Dsmvtll","Dretwd","Clsprc"]].iloc[2:]
# df1 = df1[-1000:].reset_index()
# df1.rename({'Stkcd':"Stkcd",'Trddt':"Trddt","Dsmvosd":"Dsmvosd","Dsmvtll":"Dsmvtll","Dretwd":"Dretwd","Clsprc":"Clsprc"},axis='columns',inplace=True)
df1["year"] = pd.DatetimeIndex(df1["Trddt"]).year.astype(str)
# df1["year"] = df1["year"].astype(int)
df1["year"] = df1["year"].astype(str)
df1.reset_index(drop=True,inplace=True)
df1["month"] = pd.DatetimeIndex(df1["Trddt"]).month.astype(str)
for i in range(len(df1)):
    if len(df1.loc[i,"month"]) == 1:
            df1.loc[i,"month"] = '0'+df1.loc[i,"month"]

#40 最大日回报
df2 = df1
df2["Trddt"] = df2["year"]+df2["month"]
df2 = df2[["Stkcd","Trddt","Dretwd"]].groupby(["Stkcd","Trddt"]).max().reset_index()
# df2["Dretwd"] = df2["Dretwd"].shift(1)
df2.loc[0,"Dretwd"] = None
for i in range(1,len(df2)):
    if df2.loc[i,"Stkcd"] != df2.loc[i-1,"Stkcd"]:
        df2.loc[i,"Dretwd"] = None
df2.rename({"Dretwd":"maxret"},axis='columns',inplace=True)
df2["Trddt"] = [x.date() for x in pd.to_datetime(df2["Trddt"],format='%Y%m')]
df2.to_csv('月_40.csv',index=False,encoding='utf-8_sig')

#41-44 动量
df3 = df1
df3["Trddt"] = df3["year"]+df3["month"]
#mom1m :1-month cumulative return
df3 = df3[["Stkcd","Trddt","Dretwd"]].groupby(["Stkcd","Trddt"]).sum().reset_index()
df3.rename({"Dretwd":"mom1m"},axis='columns',inplace=True)
# df3["mom1m"] = df3["mom1m"].shift(1)

#mom6m: 5-month cumulative returns ,from t-5 to t-1
df3["mom6m"] = df3["mom1m"].rolling(5).sum().shift(1)
df3["mom12m"] = df3["mom1m"].rolling(11).sum().shift(1)

df3["mom36m"] = df3["mom1m"].rolling(35).sum()
df3["mom11m"] = df3["mom1m"].rolling(11).sum()
#t-35 -> t-12 
df3["mom36m"] = (df3["mom36m"] - df3["mom11m"])
for i in range(36,len(df3)):
    if df3.loc[i,"Stkcd"] != df3.loc[i-5,"Stkcd"]:
        df3.loc[i,"mom6m"] = None
    if df3.loc[i,"Stkcd"] != df3.loc[i-11,"Stkcd"]:
        df3.loc[i,"mom12m"] = None
    if df3.loc[i,"Stkcd"] != df3.loc[i-36,"Stkcd"]:
        df3.loc[i,"mom36m"] = None
df3.drop(columns='mom11m',inplace=True)
    
#18 动量变化 chmom Cumulative returns from months t - 6 to t - 1 minus months t - 12 to t - 7
df3["chmom"] = df3["mom6m"] - (df3["mom12m"] - df3["mom6m"])
df3["Trddt"] = [x.date() for x in pd.to_datetime(df3["Trddt"],format='%Y%m')]
# df3[["Stkcd","Trddt","mon36m"]].to_csv('data/月_18_41_42_43_44_.csv',index=False,encoding='utf-8_sig')
df3.to_csv('月_18_41_42_43_44.csv',index=False,encoding='utf-8_sig')