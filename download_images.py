import os
import json
import urllib.request
from tqdm import tqdm

# coco dataset download URL template
URL_TEMPLATE = "http://images.cocodataset.org/train2017/{file_name}"

# read JSON file and extract unique image file names
with open('data/train.json', 'r') as f:
    data = json.load(f)

# collect unique image file names
image_files = set()
for item in data:
    path = item['image']['path']
    file_name = os.path.basename(path)
    image_files.add(file_name)

print(f"Need to download {len(image_files)} images")

# create directory if it doesn't exist
os.makedirs('data/images/train2017', exist_ok=True)

# download images
for file_name in tqdm(image_files):
    url = URL_TEMPLATE.format(file_name=file_name)
    save_path = f'data/images/train2017/{file_name}'
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

print("All images downloaded successfully!")