import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import DistilBertTokenizer, DistilBertModel
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

# ----------------------------- 配置 -----------------------------
CONFIG = {
    "teacher_scores_file": "output/train_scores.jsonl",
    "image_root": "",          # 图片根目录（因为jsonl中已是相对路径，留空）
    "alpha": 0.7,              # InternVL3权重
    "batch_size": 32,
    "epochs": 20,
    "lr": 2e-5,
    "max_seq_len": 64,
    "image_size": 224,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "checkpoints",
    "best_model_name": "student_distill_best.pt",
    "val_ratio": 0.1,
    "seed": 42,
}

# ----------------------------- 1. 加载教师分数 -----------------------------
def load_teacher_scores(file_path):
    """从jsonl提取 clip.prob 和 internvl.score，并保留真实标签（供后续评估用，但本脚本不评估）"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            clip_score = item.get('clip', {}).get('prob')
            internvl_score = item.get('internvl', {}).get('score')
            if clip_score is None or internvl_score is None:
                continue
            data.append({
                'image_path': item['image_path'],
                'text': item['text'],
                'clip_score': float(clip_score),
                'internvl3_score': float(internvl_score),
            })
    print(f"Loaded {len(data)} samples")
    return data

def compute_soft_labels(data, alpha):
    """加权平均融合"""
    for item in data:
        soft = alpha * item['internvl3_score'] + (1 - alpha) * item['clip_score']
        item['teacher_soft_label'] = soft
    return data

# ----------------------------- 2. 数据集 -----------------------------
class MatchDataset(Dataset):
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
        # 如果image_root非空且路径不是绝对路径，则拼接
        if self.image_root and not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.image_transform(image)
        except Exception as e:
            # 出错时用黑图代替
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
        soft_label = torch.tensor(item['teacher_soft_label'], dtype=torch.float32)
        return input_ids, attention_mask, image, soft_label

# ----------------------------- 3. 学生模型 -----------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )
    def forward(self, x):
        return self.cnn(x)

class StudentModel(nn.Module):
    def __init__(self, text_embed_dim=768, image_embed_dim=128, hidden_dim=256):
        super().__init__()
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim)
        self.image_encoder = ImageEncoder(embed_dim=image_embed_dim)
        self.image_proj = nn.Linear(image_embed_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, images):
        text_feat = self.text_encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        text_feat = self.text_proj(text_feat)
        img_feat = self.image_encoder(images)
        img_feat = self.image_proj(img_feat)
        combined = torch.cat([text_feat, img_feat], dim=1)
        return self.fc(combined).squeeze(1)

# ----------------------------- 4. 训练函数（仅含损失，无分类指标）--------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for input_ids, attn_mask, images, labels in tqdm(loader, desc="Training"):
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attn_mask, images)
        loss = F.mse_loss(outputs, labels)   # 蒸馏损失：学生输出拟合教师软标签
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    """只计算验证损失（MSE），不计算任何分类指标"""
    model.eval()
    total_loss = 0
    for input_ids, attn_mask, images, labels in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attn_mask, images)
        loss = F.mse_loss(outputs, labels)
        total_loss += loss.item()
    return total_loss / len(loader)

# ----------------------------- 5. 主程序 -----------------------------
def main():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")

    # 加载数据并融合
    raw_data = load_teacher_scores(CONFIG['teacher_scores_file'])
    data = compute_soft_labels(raw_data, CONFIG['alpha'])
    print(f"Alpha={CONFIG['alpha']}, sample soft label: {data[0]['teacher_soft_label']:.4f}")

    # 划分训练/验证
    np.random.shuffle(data)
    val_size = int(len(data) * CONFIG['val_ratio'])
    train_data = data[val_size:]
    val_data = data[:val_size]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_dataset = MatchDataset(train_data, tokenizer, transform, CONFIG['image_root'])
    val_dataset = MatchDataset(val_data, tokenizer, transform, CONFIG['image_root'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    model = StudentModel().to(device)
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_val_loss = float('inf')
    for epoch in range(1, CONFIG['epochs']+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], CONFIG['best_model_name']))
            print(f"  -> Best model saved (val_loss={val_loss:.4f})")

    print(f"Training finished. Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()