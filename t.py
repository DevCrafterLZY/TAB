import os
import re

# def update_sh_files(directory):
    # 遍历指定目录及其子目录中的所有文件
for root, dirs, files in os.walk("scripts/multivariate_detection"):
    for file in files:
        if file.endswith(".sh"):  # 确保处理的是.sh文件
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    # if '"patience": 3, ' in line:
                    #     # 替换指定的字符串
                    #     line = line.replace('"patience": 3, ', '')
                    if 'TinyTimeMixer' in line:
                        # 替换指定的字符串
                        line = line.replace('"seq_len": 96', '"seq_len": 512')
                    
                    f.write(line)
