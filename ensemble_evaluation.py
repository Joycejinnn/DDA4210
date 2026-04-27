import argparse
import os

import numpy as np
import torch

from evaluation_utils import (
    DEFAULT_CONFIG,
    compute_metrics,
    compute_precision_recall_points,
    compute_roc_points,
    create_dataloader,
    ensure_directory,
    load_student_model,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of student checkpoints.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Path to a student checkpoint. Repeat for multiple models.",
    )
    parser.add_argument("--dataset", default="data/test.json")
    parser.add_argument("--output-dir", default="evaluation_outputs/ensemble")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG["threshold"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_CONFIG["max_seq_len"])
    parser.add_argument("--image-size", type=int, default=DEFAULT_CONFIG["image_size"])
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directory(args.output_dir)
    _, loader = create_dataloader(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_student_model(path, device=device) for path in args.checkpoint]

    labels = []
    probabilities = []
    prediction_rows = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            outputs = []
            for model in models:
                outputs.append(model(input_ids, attention_mask, images).detach().cpu().numpy())
            mean_prob = np.mean(np.stack(outputs, axis=0), axis=0)
            for meta, label, probability in zip(batch["meta"], batch["label"].numpy(), mean_prob):
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

    metrics = compute_metrics(labels, probabilities, threshold=args.threshold)
    payload = {
        "dataset": args.dataset,
        "checkpoints": args.checkpoint,
        "metrics": metrics.to_dict(),
        "roc_curve": compute_roc_points(labels, probabilities),
        "precision_recall_curve": compute_precision_recall_points(labels, probabilities),
        "predictions": prediction_rows,
    }
    output_path = os.path.join(args.output_dir, "ensemble_results.json")
    save_json(payload, output_path)

    print(f"Ensemble size: {len(args.checkpoint)}")
    print(f"Accuracy:  {metrics.accuracy:.4f}")
    print(f"F1 Score:  {metrics.f1_score:.4f}")
    print(f"AUC:       {metrics.auc if metrics.auc is not None else 'N/A'}")
    print(f"Saved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
