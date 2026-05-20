import os
import json
import urllib.request
from tqdm import tqdm

# URL template for COCO 2017 training images (official source, fast)
URL_TEMPLATE = "http://images.cocodataset.org/train2017/{file_name}"

# Read the JSON file (any of train/val/test works since they point to the same images)
with open('data/train.json', 'r') as f:
    data = json.load(f)

# Collect all image file names to download
image_files = set()
for item in data:
    path = item['image']['path']
    file_name = os.path.basename(path)
    image_files.add(file_name)

print(f"Need to download {len(image_files)} images")

# Create the directory for storing images
os.makedirs('data/images/train2017', exist_ok=True)

# Download images one by one (with progress bar)
for file_name in tqdm(image_files):
    url = URL_TEMPLATE.format(file_name=file_name)
    save_path = f'data/images/train2017/{file_name}'
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

print("All images downloaded!")