# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 18:22
# @Author  : YiMing Jiang

import pandas as pd


def transfer(fileNames: list):
    """
    由于 CSMAR 提供的字段为英文，需根据说明将其修改为中文
    :param fileNames: 要修改columns名称的文件列表
    :return:
    """
    for fileName in fileNames:
        print(fileName)
        columns = []
        with open("./data/" + fileName + "字段.txt", "r", encoding="utf-8-sig") as f:
            columns = [ele.split("[")[1].split("]")[0] for ele in f.readlines()]
        df = pd.read_csv("./data/" + fileName + "(原始).csv")
        df.columns = columns
        df.to_csv("./data/" + fileName + ".csv", index=False, encoding="utf-8-sig")


def main():
    transfer(
        ["利润表", "现金流量表(间接法)", "现金流量表(直接法)", "资产负债表", "相对价值指标", "公司文件", "月个股回报率", "中国上市公司股权性质文件"])


if __name__ == "__main__":
    main()
