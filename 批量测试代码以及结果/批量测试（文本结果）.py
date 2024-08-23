import os
import requests
import pandas as pd

# 指定要上传文件的目录
directory = r"C:\Users\abc\Desktop\08月09日自动保存_1[1]"
# 服务器上传地址
upload_url = 'http://116.57.89.150:12345/upload'
# 指定Excel文件的保存路径
excel_file_path = r"C:\Users\abc\Desktop\新建 XLSX 工作表.xlsx"

# 初始化一个空的DataFrame，用于存储数据
df = pd.DataFrame(columns=['文件序号', '响应文本'])

# 计数器，用于记录文件的顺序编号
file_counter = 1

# 遍历目录中的所有文件
for filename in sorted(os.listdir(directory)):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            files = {'file': (filename, file)}
            try:
                # 发送POST请求
                response = requests.post(upload_url, files=files)
                response.raise_for_status()

                # 获取响应的文本内容
                response_text = response.text

                # 将响应文本添加到DataFrame
                df = df.append({'文件序号': file_counter, '响应文本': response_text}, ignore_index=True)

                # 更新计数器
                file_counter += 1

            except requests.exceptions.RequestException as e:
                print(f'An error occurred while uploading {filename}: {e}')

# 检查DataFrame是否为空，如果不为空，则保存到Excel文件
if not df.empty:
    # 保存Excel文件，指定engine为openpyxl，确保中文不乱码
    df.to_excel(excel_file_path, index=False, engine='openpyxl')
    print(f'Excel file has been saved to {excel_file_path}')
else:
    print('No data to save to Excel file.')