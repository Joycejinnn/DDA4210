import os
import json
import urllib.request
from tqdm import tqdm

# COCO 2017 训练集图片的 URL 模板（官方地址，速度快）
URL_TEMPLATE = "http://images.cocodataset.org/train2017/{file_name}"

# 读取 JSON 文件（train/val/test 任一个即可，因为都指向相同的图片）
with open('data/train.json', 'r') as f:
    data = json.load(f)

# 收集所有需要用到的图片文件名
image_files = set()
for item in data:
    path = item['image']['path']
    file_name = os.path.basename(path)
    image_files.add(file_name)

print(f"需要下载 {len(image_files)} 张图片")

# 创建存放图片的目录
os.makedirs('data/images/train2017', exist_ok=True)

# 逐张下载（带进度条）
for file_name in tqdm(image_files):
    url = URL_TEMPLATE.format(file_name=file_name)
    save_path = f'data/images/train2017/{file_name}'
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

print("所有图片下载完成！")