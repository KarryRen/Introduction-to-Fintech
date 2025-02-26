'''
Author: zhangwj
Date: 2024-10-29 19:52:18
LastEditTime: 2024-10-30 15:54:04
LastEditors: zhangwj
Description: 
'''
import pandas as pd
import numpy as np
import datetime as dt
import numpy as np
import datetime as dt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

trade_date_list = pd.read_csv(r"D:\vsCode\VScodeProject\pyProject\Pyproject\Courses\Fintech\Introduction-to-Fintech-DAPs\Data\trading_dates.csv")['trade_date'].astype(str).to_list()

def match_trade_date(x):
    if x in trade_date_list:
        return x
    else:
        cur = dt.datetime.strptime(x,"%Y%m%d")
        for i in range(20):
            cur = cur + dt.timedelta(1)
            cur_str = cur.strftime("%Y%m%d")
            if cur_str in trade_date_list:
                return cur_str
    return None

# 1. Data Prepare
# load text embedding 
def load_embedding(file_path):
    """读取embedding数据,保证每支股票一天只有一个embedding数据，如果某一只股票 某天有多条txet数据, 求均值即可"""
    em_df = pd.read_pickle(file_path)
    em_df['embedding'] = em_df['embedding'].apply(lambda x:eval(x))
    em_df['embedding'] = em_df['embedding'].apply(np.array)
    em_df['publishDate'] = em_df['publishDate'].astype(str)
    em_df['matched_date'] =em_df['publishDate'].apply(match_trade_date)
    embedding_df = em_df.groupby(['stockCode','matched_date'])['embedding'].mean().reset_index()
    return embedding_df


def get_train_test_XY(processed_embedding_withlable,split_date = '20240101'):
    """
    get train and test data
    """
    processed_embedding_withlable_Train = processed_embedding_withlable[processed_embedding_withlable['matched_date']<split_date]
    processed_embedding_withlable_Test = processed_embedding_withlable[processed_embedding_withlable['matched_date']>=split_date]
    X_train = np.array([i for i in processed_embedding_withlable_Train['embedding']])
    y_train = np.array(processed_embedding_withlable_Train['Label_1'])
    X_test = np.array([i for i in processed_embedding_withlable_Test['embedding']])
    y_test = np.array(processed_embedding_withlable_Test['Label_1'])
    ID_train = [i for i in processed_embedding_withlable_Train.index]
    ID_test = [i for i in processed_embedding_withlable_Test.index]
    return X_train,y_train,X_test,y_test,ID_train,ID_test

def concat_by_ID(original_df,name,data_array,ID_list):
    """
    用于根据ID列表,将降维后的data_array合并到原来的df中
    
    original_df: 已有的dataframe
    name: 新增or更新的列名
    data_array: 一个ndarray，形如（样本数,特指数）
    ID_list: get_train_test_XY切分时，每行样本的ID列表
    """
    s = pd.Series({ID_list[i]:data_array[i,:] for i in range(len(ID_list))})
    if name in original_df.columns:
        original_df[name].update(s)
    else:
        original_df[name] =   s
    return original_df

def PCA_reduction(processed_embedding_withlable,threshold):
    """
    threshold : cumulate contribution of principal component 
    return -> X_train,X_test after PCA reduction
    """
    X_train,y_train,X_test,y_test,ID_train,ID_test = get_train_test_XY(processed_embedding_withlable)
    
    # train PCA
    pca = PCA()  
    X_pca = pca.fit_transform(X_train)
    # contribution of principal component 
    explained_variance = pca.explained_variance_ratio_
    # calculate cumulate contribution
    cumulative_variance = np.cumsum(explained_variance)

    # find out the number of needed principal component to reach the threshold
    n_components = np.argmax(cumulative_variance >= threshold) + 1  
    print(f"Number of components to explain {threshold} variance: {n_components}")
    # 可视化累计方差贡献率
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.title('Cumulative Explained Variance by PCA')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(np.arange(1, len(cumulative_variance) + 1, 50))
    plt.grid()
    plt.show()
    test_X_pca = pca.transform(X_test)

    return ID_train,ID_test ,X_pca[:,:n_components],test_X_pca[:,:n_components]
    

def Lasso_reduction(processed_embedding_withlable,alpha):
    """
    alpha : L1 coeficient
    return -> X_train,X_test after Lasso reduction
    """
    X_train,y_train,X_test,y_test,ID_train,ID_test = get_train_test_XY(processed_embedding_withlable)
    
    # train Lasso
    lasso_model = Lasso(alpha=0.00001,fit_intercept =False)
    lasso_model.fit(X_train, y_train)
    feature_names =[i for i in range(0,X_train.shape[1])]
    coefficients = lasso_model.coef_
    selected_features = [feature_names[i] for i in range(len(feature_names)) if coefficients[i] != 0]
    # 计算均方误差 (MSE) 和决定系数 (R²)
    # def get_mse_r2(y, y_pred):
    #     mse = mean_squared_error(y, y_pred)
    #     r2 = r2_score(y, y_pred)
    #     print("均方误差 (MSE):", mse)
    #     print("决定系数 (R²):", r2)

    # get_mse_r2(y_train,y_pred=lasso_model.predict(X_train))
    # get_mse_r2(y_test,y_pred=lasso_model.predict(X_test))

    # 可视化
    lasso_select_df = pd.DataFrame({"Dimension":feature_names,"Coefficients":coefficients}).sort_values("Coefficients")
    plt.figure(figsize=(10, 6))
    plt.barh(lasso_select_df['Dimension'],lasso_select_df['Coefficients'],height=1)
    plt.xlabel('Coefficient Value')
    plt.title('Lasso Coefficients')
    plt.axvline(0, color='grey', lw=0.8)
    plt.show()
    return ID_train,ID_test ,X_train[:,selected_features],X_test[:,selected_features]


def XGB_reduction(processed_embedding_withlable,threshold):
    """
    threshold : cumulate importance of features
    return -> X_train,X_test after XGB reduction by features selection
    """
    X_train,y_train,X_test,y_test,ID_train,ID_test = get_train_test_XY(processed_embedding_withlable)
    
    model = xgb.XGBRegressor()  # 或者 xgb.XGBRegressor() 取决于你的任务
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': [i for i in range(X_train.shape[1])], 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    # selected_features = feature_importance[feature_importance['Importance'] > threshold]['Feature']
    importance_sum = 0
    threshold_num = 0

    for i in range(len(feature_importance)):
        importance_sum+= feature_importance.iloc[i]['Importance']
        if importance_sum > threshold:
            threshold_num = i+1
            break
    selected_features = feature_importance.iloc[:threshold_num]['Feature'].values

    return ID_train,ID_test ,X_train[:,selected_features],X_test[:,selected_features]


def add_label(horizon_list,embedding_df,DATA_DIR=r"E:\Graduatedwork\Courses\Fintech\Assignment\data"):
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
        Label.columns = ['stockCode',"matched_date",f'Label_{horizon}']
        embedding_df = pd.merge(embedding_df,Label,how='left',on= ['stockCode',"matched_date"])
    return embedding_df

# 5. 结果分析

## 5.1 不同降维模型的影响（包括不降维）

## 5.2 长期短期label的影响（短期为1,3天，长期为10,20）

if __name__ == "__main__":
    # embedding_df = load_embedding("../../../Data/text_factors/embedding.pkl")
    # embedding_df.to_pickle("../../../Data/text_factors/processed_embedding_df.pkl")
    embedding_df = pd.read_pickle("../../../Data/text_factors/processed_embedding_df.pkl")
    processed_embedding_withlable = add_label([1,3,5,10],embedding_df)
    processed_embedding_withlable.to_pickle("../../../Data/text_factors/processed_embedding_withlable.pkl")

    ## reduction
    processed_embedding_withlable = pd.read_pickle(r"D:\vsCode\VScodeProject\pyProject\Pyproject\Courses\Fintech\Introduction-to-Fintech-DAPs\Data\text_factors\processed_embedding_withlable.pkl")
    
    # PCA
    ID_train,ID_test ,X_train_PCA,X_test_PCA = PCA_reduction(processed_embedding_withlable,0.85)
    processed_embedding_withlable_PCA = concat_by_ID(processed_embedding_withlable,'PCA',X_train_PCA,ID_train)
    processed_embedding_withlable_PCA = concat_by_ID(processed_embedding_withlable_PCA,'PCA',X_test_PCA,ID_test)
    # Lasso
    ID_train,ID_test ,X_train_Lasso,X_test_Lasso = Lasso_reduction(processed_embedding_withlable,alpha = 0.00001 )
    processed_embedding_withlable_Lasso = concat_by_ID(processed_embedding_withlable,'Lasso',X_train_Lasso,ID_train)
    processed_embedding_withlable_Lasso = concat_by_ID(processed_embedding_withlable_Lasso,'Lasso',X_test_Lasso,ID_test)
    # XGB
    ID_train,ID_test ,X_train_XGB,X_test_XGB = XGB_reduction(processed_embedding_withlable,0.4)
    processed_embedding_withlable_XGB = concat_by_ID(processed_embedding_withlable,'XGB',X_train_XGB,ID_train)
    processed_embedding_withlable_XGB = concat_by_ID(processed_embedding_withlable_XGB,'XGB',X_test_XGB,ID_test)
    processed_embedding_withlable.to_pickle("processed_embedding_withlable_reduction.pkl")