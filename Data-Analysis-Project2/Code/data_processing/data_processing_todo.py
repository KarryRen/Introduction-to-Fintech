'''
Author: zhangwj
Date: 2024-10-29 19:52:18
LastEditTime: 2024-10-29 22:07:42
LastEditors: zhangwj
Description: 
'''
import pandas as pd
import numpy as np


# 1. Data Prepare
# load text embedding 
def load_embedding(file_path):
    """读取embedding数据,保证每支股票一天只有一个embedding数据，如果某一只股票 某天有多条txet数据, 求均值即可"""
    em_df = pd.read_pickle(file_path)
    em_df['embedding'] = em_df['embedding'].apply(lambda x:eval(x))
    em_df['embedding'] = em_df['embedding'].apply(np.array)
    embedding_df = em_df.groupby(['stockCode','publishDate'])['embedding'].mean().reset_index()
    return embedding_df

# dimension reduction
def PCA_reduction():
    """降维,输出形式参考load_embedding的输出"""
    pass

# 使用特征重要性那套来降维，需要依赖label
def lasso_reduction():
    """降维,输出形式参考load_embedding的输出"""
    pass

def tree_reduction():
    """降维,输出形式参考load_embedding的输出"""
    pass

def generate_label(future_horizon):
    """ 未来future_horizon天的收益率"""
    pass

def add_label():
    """ 加入label,merge的时候使用left，以embedding数据为准 """
    pass

# 5. 结果分析

## 5.1 不同降维模型的影响（包括不降维）

## 5.2 长期短期label的影响（短期为1,3天，长期为10,20）

if __name__ == "__main__":
    embedding_df = load_embedding("../../../Data/text_factors/embedding.pkl")
    embedding_df.to_pkl("../../../Data/text_factors/processed_embedding_df.pkl")