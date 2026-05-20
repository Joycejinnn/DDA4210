import os
import json
import urllib.request
from tqdm import tqdm

git add our_dataset_v2/

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


"""
Compare single distilled models and their ensemble on validation set.
Uses ground truth labels (0/1) for evaluation.
"""

import torch
import json
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer

# Import student model definition and config from training script
from train_student_distill import StudentModel, CONFIG

# ----------------------------- Configuration -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the three best models (adjust filenames if necessary)
CHECKPOINTS = {
    "alpha070": "checkpoints/student_distill_best_alpha070.pt",
    "alpha073": "checkpoints/student_distill_best_alpha073.pt",
    "alpha075": "checkpoints/student_distill_best_alpha075.pt",
}

BATCH_SIZE = 32

# ----------------------------- Dataset for Evaluation -----------------------------
class EvalDataset(Dataset):
    """Returns (input_ids, attention_mask, image, true_label) for evaluation."""
    def __init__(self, data, tokenizer, image_transform, image_root=''):
        self.data = data
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.image_root = image_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        img_path = item['image_path']
        if self.image_root and not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.image_transform(image)
        except Exception:
            # Fallback to black image if loading fails
            image = torch.zeros(3, CONFIG['image_size'], CONFIG['image_size'])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=CONFIG['max_seq_len'],
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        true_label = torch.tensor(item['true_label'], dtype=torch.float32)
        return input_ids, attention_mask, image, true_label

# ----------------------------- Data Loading with Stratified Split -----------------------------
def get_val_data():
    """Load teacher scores file, extract ground truth labels, and perform stratified split."""
    with open(CONFIG['teacher_scores_file'], 'r') as f:
        raw_data = [json.loads(line) for line in f]

    # Add true_label from ground_truth
    for item in raw_data:
        item['true_label'] = item.get('ground_truth', {}).get('label', 0)

    # Separate positive and negative samples
    pos = [item for item in raw_data if item['true_label'] == 1]
    neg = [item for item in raw_data if item['true_label'] == 0]

    val_ratio = CONFIG['val_ratio']
    total_val_size = int(len(raw_data) * val_ratio)

    # Number of positives to sample (keep class proportion)
    val_pos_size = int(len(pos) * val_ratio)
    val_neg_size = total_val_size - val_pos_size

    # Random sampling without replacement
    np.random.seed(CONFIG['seed'])
    val_pos_indices = np.random.choice(len(pos), size=val_pos_size, replace=False)
    val_neg_indices = np.random.choice(len(neg), size=val_neg_size, replace=False)

    val_data = [pos[i] for i in val_pos_indices] + [neg[i] for i in val_neg_indices]
    np.random.shuffle(val_data)

    print(f"Validation set: total={len(val_data)}, positives={val_pos_size}, negatives={val_neg_size}")
    return val_data

def get_val_loader():
    """Create DataLoader for validation set."""
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_data = get_val_data()
    dataset = EvalDataset(val_data, tokenizer, transform, CONFIG['image_root'])
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=CONFIG['num_workers'])

# ----------------------------- Evaluation Functions -----------------------------
def evaluate_model(model, loader):
    """Evaluate a single model: returns MSE and accuracy against ground truth."""
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for input_ids, attn_mask, images, label in tqdm(loader, desc="Evaluating"):
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            images = images.to(DEVICE)
            output = model(input_ids, attn_mask, images)
            preds.extend(output.cpu().numpy())
            labels.extend(label.numpy())
    preds = np.array(preds)
    labels = np.array(labels)
    mse = np.mean((preds - labels) ** 2)
    acc = np.mean((preds > 0.5) == labels)
    return mse, acc

def ensemble_predict(models, loader):
    """Average predictions from multiple models and compute metrics."""
    all_preds = []
    all_labels = []
    for input_ids, attn_mask, images, labels in tqdm(loader, desc="Ensemble"):
        input_ids = input_ids.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        batch_preds = []
        for model in models:
            output = model(input_ids, attn_mask, images)
            # Detach before converting to numpy to avoid gradient issues
            batch_preds.append(output.detach().cpu().numpy())
        avg_preds = np.mean(batch_preds, axis=0)
        all_preds.extend(avg_preds)
        all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    mse = np.mean((all_preds - all_labels) ** 2)
    acc = np.mean((all_preds > 0.5) == all_labels)
    return mse, acc

# ----------------------------- Main -----------------------------
def main():
    print("Loading validation data...")
    val_loader = get_val_loader()

    # Evaluate single models
    print("\n--- Single Model Evaluation ---")
    for name, path in CHECKPOINTS.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {name}")
            continue
        model = StudentModel().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        mse, acc = evaluate_model(model, val_loader)
        print(f"{name}: MSE = {mse:.4f}, Accuracy = {acc:.4f}")

    # Ensemble evaluation
    print("\n--- Ensemble Evaluation ---")
    models = []
    for path in CHECKPOINTS.values():
        if os.path.exists(path):
            model = StudentModel().to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            models.append(model)
    if models:
        mse_ens, acc_ens = ensemble_predict(models, val_loader)
        print(f"Ensemble (average of {len(models)} models): MSE = {mse_ens:.4f}, Accuracy = {acc_ens:.4f}")
    else:
        print("No models found for ensemble.")

if __name__ == "__main__":
    main()
