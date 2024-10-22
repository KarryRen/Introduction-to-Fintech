import akshare as ak
import pandas as pd
import os

# 获取所有股票的基础信息
stock_basic_df = ak.stock_info_a_code_name()

# 创建一个文件夹来存储所有的CSV文件
if not os.path.exists('./dividend'):
    os.makedirs('./dividend')

# 遍历所有股票代码
for index, row in stock_basic_df.iterrows():
    symbol = row['code']
    # 获取指定股票代码的数据
    
    stock_data_df = ak.stock_a_indicator_lg(symbol=symbol)
    
    # 确保trade_date列是日期格式
    stock_data_df['trade_date'] = pd.to_datetime(stock_data_df['trade_date'])
    
    # 设置筛选的开始和结束日期
    start_date = '2010-01-01'
    end_date = '2024-06-01'
    
    # 筛选数据
    filtered_df = stock_data_df[(stock_data_df['trade_date'] >= start_date) & 
                                 (stock_data_df['trade_date'] <= end_date)]
    
    # 添加Stkcd列，值为当前股票代码symbol
    filtered_df['Stkcd'] = symbol
    # 保存筛选后的数据到CSV文件
    file_path = os.path.join('dividend', f'{symbol}.csv')
    filtered_df.to_csv(file_path, index=False)

print(f'文件已保存: {file_path}')