

import os
import requests
import pandas as pd
import time  # 导入time模块用于记录时间

# 指定要上传文件的目录
directory = r"C:\Users\abc\Desktop\08月09日自动保存_1[1]"
# 服务器上传地址
upload_url = 'http://116.57.89.150:12345/upload'
# 指定Excel文件的保存路径
excel_file_path = r"C:\Users\abc\Desktop\批量测试0923完整版.xlsx"

# 初始化一个空的DataFrame，用于存储数据
df = pd.DataFrame(columns=['文件序号', '主诉', '其他', '既往病史', '现病史','音频文本','音频转文本时长', '总响应时间'])

# 计数器，用于记录文件的顺序编号
file_counter = 1

# 遍历目录中的所有文件
for filename in sorted(os.listdir(directory)):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            files = {'file': (filename, file)}
            try:
                # 记录开始时间
                start_time = time.time()

                # 发送POST请求
                response = requests.post(upload_url, files=files)
                response.raise_for_status()

                # 记录结束时间
                end_time = time.time()

                # 计算响应时间
                response_time = end_time - start_time

                # 解析JSON数据
                response_data = response.json()

                # 过滤需要的键并将数据编码转换为utf-8，除了'音频转文本时长'
                decoded_data = {}
                for key, value in response_data.items():
                    if key in df.columns[:-1]:  # 排除响应时间列
                        if key == '音频转文本时长':
                            decoded_data[key] = value  # 直接使用float类型的值
                        else:
                            decoded_data[key] = value.encode('utf-8').decode('utf-8')

                # 添加文件序号和响应时间
                decoded_data['文件序号'] = filename
                decoded_data['总响应时间'] = response_time

                # 将解码后的数据作为新的行添加到DataFrame
                df = df._append(decoded_data, ignore_index=True)

                # 更新计数器
                file_counter += 1

            except requests.exceptions.RequestException as e:
                print(f'An error occurred while uploading {filename}: {e}')
            except ValueError:
                print(f'Invalid JSON response for file {filename}')

# 检查DataFrame是否为空，如果不为空，则保存到Excel文件
if not df.empty:
    # 保存Excel文件，指定引擎为openpyxl，确保中文不乱码
    df.to_excel(excel_file_path, index=False, engine='openpyxl')
    print(f'Excel file has been saved to {excel_file_path}')
else:
    print('No data to save to Excel file.')