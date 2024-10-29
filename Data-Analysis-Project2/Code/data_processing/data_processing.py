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
def load_embedding(file_path,):
    """读取embedding数据,保证每支股票一天只有一个embedding数据，如果某一只股票 某天有多条txet数据, 求均值即可"""
    pass

# dimension reduction
def PCA_reduction():
    pass

# 使用特征重要性那套来降维，需要依赖label
def lasso_reduction():
    pass

def tree_reduction():
    pass

def generate_label(future_horizon):
    """ 未来future_horizon天的收益率"""
    pass

def add_label():
    """ 加入label,merge的时候使用left，以embedding数据为准 """

def add_digit_data():
    """ 在embedding数据上,加入部分量价数据（基于重要性选择，merge的时候使用left，以embedding数据为准 """
    pass

# 切割训练测试集
def split_data():
    pass

# 2. Modeling & Prediction（分类模型）

def linear():
    pass
def XGboost():
    pass
def lstm():
    pass
# 3. 评估函数
def F1_accuracy_recall():
    pass


# 4. 运行函数

def run(model,label_horizon,dimension_reduction_method,mode='train'):
    """
    model:使用的模型
    label_horizon: label是未来多少天的收益率
    dimension_reduction_method：降维方法
    mode：训练还是测试
    
    return ->评估函数的一些指标
    """
    pass


# 5. 结果分析
## 5.1 加入量价数据的影响

## 5.2 不同降维模型的影响（包括不降维）

## 5.3 长期短期label的影响（短期为1,3天，长期为 ）
