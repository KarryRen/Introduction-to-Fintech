# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : YiMing Jiang

from itertools import product
import pandas as pd
import os


def complete_code(code):
    """
    代码格式处理
    :param code: 股票代码的int值
    :return: 整理后的code字符串
    """
    code = str(code)
    if len(code) < 6:
        code = "0" * (6 - len(code)) + code + ".sz"
    elif code[:2] == "60":
        code += ".sh"
    return code


def time_format(time):
    """
    时间格式处理
    :param time: yyyy-mm-dd格式的日期字符串
    :return: yyyymmdd格式的日期字符串
    """
    time = pd.to_datetime(time)
    year, month, day = str(time.year), str(time.month), str(time.day)
    return year + "0" * (2 - len(month)) + str(time.month) + "0" * (2 - len(day)) + str(time.day)


def refine(fileNames: list):
    """
    根据文件名序列对因子格式进行处理，以符合格式要求
    :param fileNames: 要处理的文件序列
    :return: Null
    """
    code = pd.read_csv("./data/stock_code.csv")["Code"].values
    for fileName in fileNames:
        print(fileName)
        df = pd.read_csv("./factor/" + fileName + ".csv", encoding="utf-8-sig")
        # 补全代码
        df["证券代码"] = df["证券代码"].apply(lambda x: complete_code(x))
        # 筛选合规股票
        df = df[df["证券代码"].isin(code)]
        # 调整时间格式
        df["统计截止日期"] = df["统计截止日期"].apply(lambda x: time_format(x))
        df.columns = ["Code", "Date", fileName]
        df.to_csv("./refined_format_factor/" + fileName + ".csv", index=False, encoding="utf-8-sig")


def extend(fileNames: list):
    """
    根据要求调整格式, index为交易日，columns为股票，value为因子值
    :param fileNames: 要处理的因子值列表
    :return: Null
    """
    # 读取要处理的股票代码
    stockCode = pd.read_csv("./data/stock_code.csv", encoding="utf-8-sig")
    # 读取2010.01.01-2024.06.01的交易日序列
    tradeDate = pd.read_csv("./data/trading_dates.csv", encoding="utf-8-sig")
    # 为与根据财报计算出的因子合并，暂时将模板的日期扩充为交易日序列 + 季末
    quarterDate = list(product([str(e) for e in range(2010, 2024)], ["0331", "0630", "0930", "1231"]))
    quarterDate.extend([("2009", "1231"), ("2024", "0331")])
    quarterDate = [e[0] + e[1] for e in quarterDate]

    firstDay = list(product([str(e) for e in range(2010, 2024)], ["0101", "0201", "0301", "0401", "0501", "0601",
                                                                  "0701", "0801", "0901", "1001", "1101", "1201"]))
    firstDay.extend([("2024", "0101"), ("2024", "0201"), ("2024", "0301"), ("2024", "0401"), ("2024", "0501"), ("2024", "0601")])
    firstDay = [e[0] + e[1] for e in firstDay]
    totalDate = set(tradeDate["trade_date"].astype(str).values)
    totalDate.update(quarterDate)
    totalDate.update(firstDay)
    for fileName in fileNames:
        # 准备空白模板
        df_template = pd.DataFrame(index=totalDate, columns=stockCode["Code"].values)
        df_template.sort_index(inplace=True)
        print(fileName)
        df = pd.read_csv("./refined_format_factor/" + fileName + ".csv")
        df["Date"] = df["Date"].astype(str)
        df_group = df.groupby("Code")
        # 逐股票处理
        for code, df_code in df_group:
            df_code.set_index("Date", inplace=True)
            # 这里合并时只能是left，因为返回值要赋给df_template[code]，行数不能变
            df_template[code] = pd.merge(df_template[code], df_code, left_index=True, right_on="Date", how="left")[fileName].values
        df_template.fillna(method="ffill", inplace=True)
        df_template.fillna(0, inplace=True)

        df_template.insert(loc=0, column="Date", value=df_template.index)
        # 去除非交易日的季末
        df_template = df_template[df_template["Date"].isin(tradeDate["trade_date"].astype(str).values)]
        df_template.to_csv("./extended_factor/" + fileName + ".csv", index=False, encoding="utf-8-sig")


def merge_table(fileNames: list):
    df = pd.read_csv(r"./extended_factor//" + fileNames[0] + ".csv", encoding="utf-8-sig")
    df = pd.DataFrame(df.set_index("Date").unstack(), columns=[fileNames[0]])
    for fileName in fileNames[1:]:
        print(fileName)
        df_tmp = pd.read_csv(r"./extended_factor//" + fileName + ".csv", encoding="utf-8-sig")
        df_tmp = pd.DataFrame(df_tmp.set_index("Date").unstack(), columns=[fileName])
        df = pd.merge(df, df_tmp, left_index=True, right_index=True)
    df.to_csv(r"./extended_factor//Factor.csv", index=True, encoding="utf-8-sig")


def main():
    li = os.listdir(r"F:\Fintech_Project1\factor")
    fileNames = [ele.split(".")[0] for ele in li]
    fileNames.sort(key=lambda x: int(x.split("_")[1]))
    print("refine")
    refine(fileNames)
    print("extend")
    extend(fileNames)
    print("merge")
    merge_table(fileNames)


if __name__ == "__main__":
    main()
