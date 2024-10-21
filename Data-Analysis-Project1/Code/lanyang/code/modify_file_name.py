import os 
import pandas as pd

file_path = "/Users/lanyang/Desktop/Machine-Learning-in-the-Chinese-Stock-Market-Reproduction-main/lanyang"

file_names = [f for f in os.listdir(file_path) if f.startswith('data\\') and f.endswith('.csv')]

data_frames = []

for file_name in file_names:
    full_file_path = os.path.join (file_path, file_name)
    # 删除文件名中的 'data_' 前缀
    new_file_name = file_name[5:]  # 假设 'data_' 长度为 5
    new_file_path = os.path.join(file_path, new_file_name)
    os.rename(full_file_path, new_file_path)
    print(f"Renamed '{file_name}' to '{new_file_name}'")