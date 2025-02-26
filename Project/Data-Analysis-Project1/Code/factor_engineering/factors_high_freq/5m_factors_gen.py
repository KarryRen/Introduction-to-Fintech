'''
Author: zhangwj
Date: 2024-10-29 19:25:27
LastEditTime: 2024-10-29 19:33:35
LastEditors: zhangwj
'''
import pandas as pd
DATA_DIR = r".\data"
volume = pd.read_hdf(f'{DATA_DIR}/volume.h5', key='df')
turnover = pd.read_hdf(f'{DATA_DIR}/turnover_float.h5', key='df')
amount = pd.read_hdf(f'{DATA_DIR}/amount.h5', key='df')
close = pd.read_hdf(f'{DATA_DIR}/close.h5', key='df')
high = pd.read_hdf(f'{DATA_DIR}/high.h5', key='df')
low = pd.read_hdf(f'{DATA_DIR}/low.h5', key='df')

# base factors 
imbalance = (high-low)/(high+low)
price_spread = high-low
price_ratio = low/high
vwap = amount/volume

### moment_alpha : std,skwness,kurtosise,and lagged 1 autocorrelation
def get_alpha(data,name):
    def moment_alpha(data,name):
        # function to get std,skwness, and kurtosise
        groups = data.groupby(data.index.date)
        data_std = groups.std().unstack()
        data_skew = groups.skew().unstack()
        kurt_result ={}
        for date,i in groups:
            kurt_result[date] = i.kurt()
        data_kurt = pd.DataFrame(kurt_result)
        data_kurt = data_kurt.T.unstack()
        data_autocorr = groups.corrwith(groups.shift(1)).unstack()
        result = pd.concat([data_std,data_skew,data_kurt,data_autocorr],axis=1)
        result.columns = [name+'_'+i for i in ['std','skew','kurt','autocorr']]
        return  result
    data_alpha = moment_alpha(data,name)
    # morning session
    data_am = data.between_time(start_time='09:30',end_time='11:30')
    # afternoon session
    data_pm = data.between_time(start_time='13:00',end_time='15:00')
    data_am_alpha = moment_alpha(data_am,f'{name}_am')
    data_pm_alpha = moment_alpha(data_pm,f'{name}_pm')
    # spread between morning and afternoon session
    data_diff_alpha = data_am_alpha.rename({i: i.replace('_am_','_diff_') for i in data_am_alpha.columns},axis=1)- data_pm_alpha.rename({i: i.replace('pm','diff') for i in data_pm_alpha.columns},axis=1)
    all_data_factor = pd.concat([data_alpha,data_am_alpha,data_pm_alpha,data_diff_alpha],axis=1)

    return all_data_factor
# traverse to get factors
for name,data in zip(["volume","turnover","amount","close","imbalance","price_spread","price_ratio","vwap"],[volume,turnover,amount,close,imbalance,price_spread,price_ratio,vwap]):
    all_factor = get_alpha(data,name)
    all_factor.to_hdf(f"./alphas_df/all_{name}_factor.h5", key='df', mode='w')

