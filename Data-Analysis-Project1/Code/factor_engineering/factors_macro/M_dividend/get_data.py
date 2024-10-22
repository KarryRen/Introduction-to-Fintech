import akshare as ak
import pandas as pd
import os
import time

# 读取 stockcode.csv 文件中的股票代码
stock_code_path = '/Users/lanyang/Desktop/Machine-Learning-in-the-Chinese-Stock-Market-Reproduction-main/lanyang/code/M_dividend/stockcode.csv'
stock_codes_df = pd.read_csv(stock_code_path, header=None, names=['Stkcd'])
stock_codes = stock_codes_df['Stkcd'].tolist()

# 创建一个文件夹来存储所有的CSV文件
if not os.path.exists('./dividend1'):
    os.makedirs('./dividend1')

# 遍历股票代码列表
for symbol in stock_codes:
    symbol = symbol.split('.')[0]
    # time.sleep(2)
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
    file_path = os.path.join('./dividend1', f'{symbol}.csv')
    filtered_df.to_csv(file_path, index=False)

print(f'文件已保存到: {file_path}')