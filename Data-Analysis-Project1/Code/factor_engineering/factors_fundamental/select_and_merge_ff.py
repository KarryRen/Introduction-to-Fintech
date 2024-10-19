# -*- coding: utf-8 -*-
# @Time    : 2024/10/19 18:49
# @Author  : Karry Ren

""" Select and merge fundamental factors of YiMing and LanYang. """

import pandas as pd

data_root_path = "../../../../Data"

# ---- Read YiMing's Factors ---- #
yiming_factor_list = [
    "Factor_1", "Factor_2", "Factor_6", "Factor_10", "Factor_11", "Factor_18",
    "Factor_30", "Factor_31", "Factor_32", "Factor_38", "Factor_39", "Factor_46",
    "Factor_47", "Factor_48", "Factor_50", "Factor_52", "Factor_60", "Factor_81",
    "Factor_91", "Factor_92", "Factor_93", "Factor_94"
]
yiming_df = pd.read_csv(f"{data_root_path}/fundamental_factors/yiming_factors.csv")
print(yiming_df)

# ---- Read LanYang's Factors ---- #
# lanyang_factor_list = [
#
# ]
# lanyang_df1 = pd.read_csv(f"{data_root_path}/fundamental_factors/lanyang/merged_file1.csv")
# print(lanyang_df1)
