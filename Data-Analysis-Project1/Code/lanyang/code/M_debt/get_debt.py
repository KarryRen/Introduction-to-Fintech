import akshare as ak

bond_zh_us_rate_df = ak.bond_zh_us_rate(start_date="19901219")
file_path = "/Users/lanyang/Desktop/Machine-Learning-in-the-Chinese-Stock-Market-Reproduction-main/lanyang/code/M_debt/debt_new.csv"
bond_zh_us_rate_df.to_csv(file_path, index=False)