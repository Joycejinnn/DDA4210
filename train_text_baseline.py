import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from evaluation_utils import (
    DEFAULT_CONFIG,
    TextOnlyBaselineModel,
    compute_metrics,
    create_text_dataloader,
    ensure_directory,
    predict_text_probabilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text-only DistilBERT baseline on ground-truth labels.")
    parser.add_argument("--train-dataset", default="data/train.json")
    parser.add_argument("--eval-dataset", default="data/test.json")
    parser.add_argument("--output-dir", default="evaluation_outputs/text_baseline")
    parser.add_argument("--checkpoint-name", default="text_only_baseline.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_CONFIG["max_seq_len"])
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG["threshold"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text-model-name",
        default=DEFAULT_CONFIG["text_model_name"],
        help="Hugging Face model name or local directory for DistilBERT tokenizer/model files.",
    )
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = F.binary_cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device, threshold: float) -> dict:
    labels, probabilities, rows = predict_text_probabilities(model, loader, device)
    metrics = compute_metrics(labels, probabilities, threshold=threshold)
    return {
        "metrics": metrics,
        "labels": labels,
        "probabilities": probabilities,
        "predictions": rows,
    }


def main() -> None:
    args = parse_args()
    ensure_directory(args.output_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _, train_loader = create_text_dataloader(
        dataset_path=args.train_dataset,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        text_model_name=args.text_model_name,
        shuffle=True,
    )
    _, eval_loader = create_text_dataloader(
        dataset_path=args.eval_dataset,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        text_model_name=args.text_model_name,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextOnlyBaselineModel(text_model_name=args.text_model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_state = None
    best_f1 = -1.0
    best_summary = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        summary = evaluate(model, eval_loader, device, threshold=args.threshold)
        f1 = summary["metrics"].f1_score
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"acc={summary['metrics'].accuracy:.4f}, f1={f1:.4f}, auc={summary['metrics'].auc}"
        )
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_summary = summary

    checkpoint_path = os.path.join(args.output_dir, args.checkpoint_name)
    torch.save(best_state, checkpoint_path)
    print(f"Saved best checkpoint to: {checkpoint_path}")
    print(
        f"Best metrics: acc={best_summary['metrics'].accuracy:.4f}, "
        f"f1={best_summary['metrics'].f1_score:.4f}, auc={best_summary['metrics'].auc}"
    )


if __name__ == "__main__":
    main()
