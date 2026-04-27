import argparse
import os

from evaluation_utils import (
    DEFAULT_CONFIG,
    compute_metrics,
    compute_precision_recall_points,
    compute_roc_points,
    create_dataloader,
    ensure_directory,
    load_student_model,
    predict_probabilities,
    save_json,
)
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a single distilled student model.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt student checkpoint.")
    parser.add_argument("--dataset", default="data/test.json", help="Evaluation dataset JSON.")
    parser.add_argument("--output-dir", default="evaluation_outputs/single_model", help="Output directory.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG["threshold"], help="Decision threshold.")
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
    model = load_student_model(args.checkpoint, device=device)
    labels, probabilities, prediction_rows = predict_probabilities(model, loader, device=device)
    metrics = compute_metrics(labels, probabilities, threshold=args.threshold)

    payload = {
        "model_name": os.path.basename(args.checkpoint),
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "metrics": metrics.to_dict(),
        "roc_curve": compute_roc_points(labels, probabilities),
        "precision_recall_curve": compute_precision_recall_points(labels, probabilities),
        "predictions": prediction_rows,
    }

    output_path = os.path.join(args.output_dir, "single_model_results.json")
    save_json(payload, output_path)

    print(f"Model: {payload['model_name']}")
    print(f"Dataset: {args.dataset}")
    print(f"Accuracy:  {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"F1 Score:  {metrics.f1_score:.4f}")
    print(f"AUC:       {metrics.auc if metrics.auc is not None else 'N/A'}")
    print(f"Saved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
