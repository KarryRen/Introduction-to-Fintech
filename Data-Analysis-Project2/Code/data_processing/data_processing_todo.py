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

def add_label(horizon_list,DATA_DIR=r'"E:\Graduatedwork\Courses\Fintech\Assignment\data"'):
    """ 
    horizon_list: list contains length of future days to calcalate return as lable 
    """
    volume = pd.read_hdf(f'{DATA_DIR}/volume.h5', key='df')
    amount = pd.read_hdf(f'{DATA_DIR}/amount.h5', key='df')
    volume = volume[volume.index>=pd.to_datetime("2019-01-01")]
    amount = amount[amount.index>=pd.to_datetime("2019-01-01")]
    backadj = pd.read_hdf(f'{DATA_DIR}/back_adj.h5', key='df')
    backadj = backadj[backadj.index>=pd.to_datetime("2019-01-01")]
    backadj = backadj.groupby(backadj.index.date).mean()

    def part_sum(df ):
        res = df.between_time('09:30',"09:40")
        res = res.groupby(res.index.date).sum()
        return res
    trade_vwap = part_sum(amount)/part_sum(volume) *backadj 
    for horizon in horizon_list:
        daily_return = trade_vwap.shift(-horizon)/trade_vwap - 1
        Label = daily_return.shift(1)
        Label.index = Label.index.astype(str).str.replace('-','')
        Label.columns = Label.columns.str.lower()
        Label = Label.unstack()
        Label = Label.reset_index()
        Label.columns = ['stockCode',"publishDate",f'Label_{horizon}']
        embedding_df = pd.merge(embedding_df,Label,how='left',on= ['stockCode',"publishDate"])
    return embedding_df

# 5. 结果分析

## 5.1 不同降维模型的影响（包括不降维）

## 5.2 长期短期label的影响（短期为1,3天，长期为10,20）

if __name__ == "__main__":
    embedding_df = load_embedding("../../../Data/text_factors/embedding.pkl")
    embedding_df.to_pickle("../../../Data/text_factors/processed_embedding_df.pkl")