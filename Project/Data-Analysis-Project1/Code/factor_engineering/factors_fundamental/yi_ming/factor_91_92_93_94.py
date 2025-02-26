# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : YiMing Jiang

import pandas as pd

EN = pd.read_csv("./data/中国上市公司股权性质文件.csv", encoding="utf-8-sig")
EN = EN.iloc[2:]

EN["soe"], EN["private"], EN["foreign"], EN["others"] = 0, 0, 0, 0
EN["股权性质编码"] = EN["股权性质编码"].fillna(0)
EN.groupby("股权性质编码")["股权性质编码"].count()


def My_contain(Set, Col, Key, FillCol, Num):
    Set.loc[Set[Col].str.contains(Key, na=False), FillCol] = Num


Set = EN
Col = "股权性质编码"
Num = 1
My_contain(Set, Col, "1", "soe", Num)
My_contain(Set, Col, "2", "private", Num)
My_contain(Set, Col, "3", "foreign", Num)
My_contain(Set, Col, "4", "others", Num)
My_contain(Set, Col, "0", "others", Num)

EN_ = EN[["证券代码", "截止日期", "soe", "private", "foreign", "others"]]

EN_["截止日期"] = pd.to_datetime(EN_["截止日期"], format="%Y-%m-%d")
EN_["统计截止日期"] = EN_["截止日期"].dt.year
EN_ = EN_[["证券代码", "统计截止日期", "soe", "private", "foreign", "others"]].reset_index(drop=True)
EN_.rename({"soe": "Factor_91"}, axis="columns", inplace=True)
EN_.rename({"private": "Factor_92"}, axis="columns", inplace=True)
EN_.rename({"foreign": "Factor_93"}, axis="columns", inplace=True)
EN_.rename({"others": "Factor_94"}, axis="columns", inplace=True)
EN_["统计截止日期"] = EN_["统计截止日期"].apply(lambda x: str(x) + "-01-01")

df_soe = EN_[["证券代码", "统计截止日期", "Factor_91"]]
df_soe.to_csv("./factor/Factor_91.csv", index=False, encoding="utf-8-sig")

df_soe = EN_[["证券代码", "统计截止日期", "Factor_92"]]
df_soe.to_csv("./factor/Factor_92.csv", index=False, encoding="utf-8-sig")

df_soe = EN_[["证券代码", "统计截止日期", "Factor_93"]]
df_soe.to_csv("./factor/Factor_93.csv", index=False, encoding="utf-8-sig")

df_soe = EN_[["证券代码", "统计截止日期", "Factor_94"]]
df_soe.to_csv("./factor/Factor_94.csv", index=False, encoding="utf-8-sig")
