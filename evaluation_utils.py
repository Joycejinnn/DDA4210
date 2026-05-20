import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import DistilBertModel, DistilBertTokenizer
import torchvision.transforms as transforms


DEFAULT_CONFIG = {
    "max_seq_len": 64,
    "image_size": 224,
    "num_workers": 0,
    "batch_size": 32,
    "threshold": 0.5,
    "text_model_name": "distilbert-base-uncased",
}


def flatten_dataset_records(data: Sequence[dict]) -> List[dict]:
    samples: List[dict] = []
    for item in data:
        image_info = item.get("image", {})
        image_path = image_info.get("path") or item.get("image_path")
        if not image_path:
            continue
        sample_id = item.get("id")
        for text in item.get("positive", []):
            if text:
                samples.append(
                    {
                        "id": sample_id,
                        "image_path": image_path,
                        "text": text,
                        "label": 1,
                    }
                )
        for text in item.get("negative", []):
            if text:
                samples.append(
                    {
                        "id": sample_id,
                        "image_path": image_path,
                        "text": text,
                        "label": 0,
                    }
                )
    return samples


def load_flattened_samples(dataset_path: str) -> List[dict]:
    with open(dataset_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return flatten_dataset_records(data)


def load_teacher_score_records(score_file: str) -> List[dict]:
    records = []
    with open(score_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_record_key(image_path: str, text: str) -> str:
    return f"{image_path}::{text}"


def align_teacher_records_to_dataset(samples: Sequence[dict], records: Sequence[dict]) -> List[dict]:
    record_map = {
        build_record_key(record["image_path"], record["text"]): record
        for record in records
        if record.get("image_path") and record.get("text")
    }
    aligned = []
    for sample in samples:
        key = build_record_key(sample["image_path"], sample["text"])
        record = record_map.get(key)
        if record is not None:
            aligned.append(record)
    return aligned


@dataclass
class MetricBundle:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: Optional[float]
    threshold: float
    confusion_matrix: List[List[int]]
    positive_count: int
    negative_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc": self.auc,
            "threshold": self.threshold,
            "confusion_matrix": self.confusion_matrix,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }


def compute_metrics(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5
) -> MetricBundle:
    y_true = np.asarray(labels, dtype=np.int32)
    y_prob = np.asarray(probabilities, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int32)

    auc = None
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return MetricBundle(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
        auc=auc,
        threshold=threshold,
        confusion_matrix=matrix,
        positive_count=int(np.sum(y_true == 1)),
        negative_count=int(np.sum(y_true == 0)),
    )


def compute_roc_points(labels: Sequence[int], probabilities: Sequence[float]) -> Optional[dict]:
    y_true = np.asarray(labels, dtype=np.int32)
    y_prob = np.asarray(probabilities, dtype=np.float32)
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def compute_precision_recall_points(
    labels: Sequence[int], probabilities: Sequence[float]
) -> Optional[dict]:
    y_true = np.asarray(labels, dtype=np.int32)
    y_prob = np.asarray(probabilities, dtype=np.float32)
    if len(np.unique(y_true)) < 2:
        return None
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


class MatchEvaluationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[dict],
        tokenizer: DistilBertTokenizer,
        image_transform: transforms.Compose,
        max_seq_len: int,
    ) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = self._load_image(sample["image_path"])
        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "meta": sample,
        }

    def _load_image(self, image_path: str) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert("RGB")
            return self.image_transform(image)
        except Exception:
            size = self.image_transform.transforms[0].size[0]
            return torch.zeros(3, size, size)


def collate_samples(batch: Sequence[dict]) -> dict:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "meta": [item["meta"] for item in batch],
    }


class TextOnlyDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[dict],
        tokenizer: DistilBertTokenizer,
        max_seq_len: int,
    ) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "meta": sample,
        }


def collate_text_samples(batch: Sequence[dict]) -> dict:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "meta": [item["meta"] for item in batch],
    }


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.cnn(images)


class StudentModel(nn.Module):
    def __init__(
        self,
        text_embed_dim: int = 768,
        image_embed_dim: int = 128,
        hidden_dim: int = 256,
        text_model_name: str = DEFAULT_CONFIG["text_model_name"],
    ) -> None:
        super().__init__()
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim)
        self.image_encoder = ImageEncoder(embed_dim=image_embed_dim)
        self.image_proj = nn.Linear(image_embed_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor
    ) -> torch.Tensor:
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_proj(text_output.last_hidden_state[:, 0, :])
        image_feat = self.image_proj(self.image_encoder(images))
        combined = torch.cat([text_feat, image_feat], dim=1)
        return self.fc(combined).squeeze(1)


class TextOnlyBaselineModel(nn.Module):
    def __init__(
        self,
        text_embed_dim: int = 768,
        hidden_dim: int = 256,
        text_model_name: str = DEFAULT_CONFIG["text_model_name"],
    ) -> None:
        super().__init__()
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = text_output.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding).squeeze(1)


def create_dataloader(
    dataset_path: str,
    batch_size: int = DEFAULT_CONFIG["batch_size"],
    max_seq_len: int = DEFAULT_CONFIG["max_seq_len"],
    image_size: int = DEFAULT_CONFIG["image_size"],
    num_workers: int = DEFAULT_CONFIG["num_workers"],
    text_model_name: str = DEFAULT_CONFIG["text_model_name"],
) -> Tuple[List[dict], DataLoader]:
    samples = load_flattened_samples(dataset_path)
    tokenizer = DistilBertTokenizer.from_pretrained(text_model_name)
    image_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = MatchEvaluationDataset(samples, tokenizer, image_transform, max_seq_len=max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )
    return samples, loader


def create_text_dataloader(
    dataset_path: str,
    batch_size: int = DEFAULT_CONFIG["batch_size"],
    max_seq_len: int = DEFAULT_CONFIG["max_seq_len"],
    num_workers: int = DEFAULT_CONFIG["num_workers"],
    text_model_name: str = DEFAULT_CONFIG["text_model_name"],
    shuffle: bool = False,
) -> Tuple[List[dict], DataLoader]:
    samples = load_flattened_samples(dataset_path)
    tokenizer = DistilBertTokenizer.from_pretrained(text_model_name)
    dataset = TextOnlyDataset(samples, tokenizer, max_seq_len=max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_text_samples,
    )
    return samples, loader


def load_student_model(
    checkpoint_path: str,
    device: torch.device,
    text_model_name: str = DEFAULT_CONFIG["text_model_name"],
) -> StudentModel:
    model = StudentModel(text_model_name=text_model_name)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_text_only_model(
    checkpoint_path: str,
    device: torch.device,
    text_model_name: str = DEFAULT_CONFIG["text_model_name"],
) -> TextOnlyBaselineModel:
    model = TextOnlyBaselineModel(text_model_name=text_model_name)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_probabilities(
    model: StudentModel, loader: DataLoader, device: torch.device
) -> Tuple[List[int], List[float], List[dict]]:
    labels: List[int] = []
    probabilities: List[float] = []
    prediction_rows: List[dict] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            outputs = model(input_ids, attention_mask, images).detach().cpu().numpy()

            for meta, label, probability in zip(batch["meta"], batch["label"].numpy(), outputs):
                labels.append(int(label))
                probabilities.append(float(probability))
                prediction_rows.append(
                    {
                        "id": meta.get("id"),
                        "image_path": meta["image_path"],
                        "text": meta["text"],
                        "label": int(label),
                        "probability": float(probability),
                    }
                )

    return labels, probabilities, prediction_rows


def predict_text_probabilities(
    model: TextOnlyBaselineModel, loader: DataLoader, device: torch.device
) -> Tuple[List[int], List[float], List[dict]]:
    labels: List[int] = []
    probabilities: List[float] = []
    prediction_rows: List[dict] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask).detach().cpu().numpy()

            for meta, label, probability in zip(batch["meta"], batch["label"].numpy(), outputs):
                labels.append(int(label))
                probabilities.append(float(probability))
                prediction_rows.append(
                    {
                        "id": meta.get("id"),
                        "image_path": meta["image_path"],
                        "text": meta["text"],
                        "label": int(label),
                        "probability": float(probability),
                    }
                )

    return labels, probabilities, prediction_rows


def compute_teacher_baseline(
    samples: Sequence[dict],
    score_file: str,
    source: str,
    alpha: Optional[float] = None,
    threshold: float = 0.5,
) -> Tuple[MetricBundle, List[dict]]:
    records = align_teacher_records_to_dataset(samples, load_teacher_score_records(score_file))
    labels: List[int] = []
    probabilities: List[float] = []
    rows: List[dict] = []

    for record in records:
        label = record.get("ground_truth", {}).get("label")
        clip_prob = record.get("clip", {}).get("prob")
        internvl_score = record.get("internvl", {}).get("score")

        if label is None:
            continue

        if source == "clip":
            probability = clip_prob
        elif source == "internvl":
            probability = internvl_score
        elif source == "fusion":
            if alpha is None or clip_prob is None or internvl_score is None:
                continue
            probability = alpha * float(internvl_score) + (1.0 - alpha) * float(clip_prob)
        else:
            raise ValueError(f"Unsupported teacher baseline source: {source}")

        if probability is None:
            continue

        labels.append(int(label))
        probabilities.append(float(probability))
        rows.append(
            {
                "id": record.get("id"),
                "image_path": record["image_path"],
                "text": record["text"],
                "label": int(label),
                "probability": float(probability),
            }
        )

    return compute_metrics(labels, probabilities, threshold=threshold), rows


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(payload: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
