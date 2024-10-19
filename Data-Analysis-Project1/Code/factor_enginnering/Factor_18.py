import pandas as pd

df = pd.read_csv("./data/月个股回报率.csv", encoding="utf-8-sig")
df["shift_1"] = (df["考虑现金红利再投资的月个股回报率"].shift(1) + 1).cumprod()
df["shift_6"] = (df["考虑现金红利再投资的月个股回报率"].shift(6) + 1).cumprod()
df["shift_7"] = (df["考虑现金红利再投资的月个股回报率"].shift(7) + 1).cumprod()
df["shift_12"] = (df["考虑现金红利再投资的月个股回报率"].shift(12) + 1).cumprod()
df["Factor_18"] = (df["shift_1"] / df["shift_6"]) - (df["shift_7"] / df["shift_12"])
df = df.dropna(subset=["Factor_18"])
df["统计截止日期"] = df["交易月份"].apply(lambda x: x + "-01")
df["统计截止日期"] = pd.to_datetime(df["统计截止日期"], format="%Y-%m-%d")

df_chmon = df[["证券代码", "统计截止日期", "Factor_18"]]
df_chmon.to_csv('./factor/Factor_18.csv', index=False, encoding='utf-8-sig')
