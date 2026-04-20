"""
Ensemble evaluation using ground truth labels.
Loads several checkpoints, averages their predictions on the validation set,
and reports MSE (pred vs true label) and accuracy.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from train_student_distill import StudentModel, CONFIG
from transformers import DistilBertTokenizer
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import json
import os
from PIL import Image

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINTS = [
    "checkpoints/student_distill_best_alpha070.pt",
    "checkpoints/student_distill_best_alpha073.pt",
    "checkpoints/student_distill_best_alpha075.pt",
]
BATCH_SIZE = 32

class EvalDataset(Dataset):
    """Dataset that returns (input_ids, attention_mask, image, true_label)."""
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

def get_val_data():
    """Load validation data with ground truth labels, same split as training."""
    # Load teacher scores file (contains ground_truth.label)
    with open(CONFIG['teacher_scores_file'], 'r') as f:
        raw_data = [json.loads(line) for line in f]
    # Add true_label field from ground_truth.label
    for item in raw_data:
        item['true_label'] = item.get('ground_truth', {}).get('label', 0)
    # Same random seed and split as training
    np.random.seed(CONFIG['seed'])
    np.random.shuffle(raw_data)
    val_size = int(len(raw_data) * CONFIG['val_ratio'])
    val_data = raw_data[:val_size]
    return val_data

def get_val_loader():
    """Create validation DataLoader."""
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_data = get_val_data()
    dataset = EvalDataset(val_data, tokenizer, transform, CONFIG['image_root'])
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=CONFIG['num_workers'])

def load_models(checkpoint_paths):
    models = []
    for path in checkpoint_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist, skipping.")
            continue
        model = StudentModel().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models.append(model)
        print(f"Loaded {path}")
    return models

@torch.no_grad()
def ensemble_predict(models, loader):
    all_preds = []
    all_labels = []
    for input_ids, attn_mask, images, labels in tqdm(loader, desc="Ensemble"):
        input_ids = input_ids.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)  # true labels (0/1)
        batch_preds = []
        for model in models:
            outputs = model(input_ids, attn_mask, images)
            batch_preds.append(outputs.cpu().numpy())
        avg_preds = np.mean(batch_preds, axis=0)
        all_preds.extend(avg_preds)
        all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

def compute_metrics(preds, labels):
    mse = np.mean((preds - labels) ** 2)
    pred_binary = (preds > 0.5).astype(int)
    acc = np.mean(pred_binary == labels)
    return mse, acc

def main():
    print("Loading models...")
    models = load_models(CHECKPOINTS)
    if not models:
        print("No models loaded. Please check checkpoint paths.")
        return
    print(f"Successfully loaded {len(models)} models")
    
    print("Loading validation data with ground truth labels...")
    val_loader = get_val_loader()
    
    print("Running ensemble prediction...")
    preds, labels = ensemble_predict(models, val_loader)
    
    mse, acc = compute_metrics(preds, labels)
    print(f"Ensemble results (vs ground truth): MSE = {mse:.4f}, Accuracy = {acc:.4f}")
    
    np.save("ensemble_preds.npy", preds)
    np.save("ensemble_labels.npy", labels)

if __name__ == "__main__":
    main()