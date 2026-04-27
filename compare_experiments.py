import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from evaluation_utils import (
    DEFAULT_CONFIG,
    StudentModel,
    compute_metrics,
    compute_precision_recall_points,
    compute_roc_points,
    compute_teacher_baseline,
    create_dataloader,
    ensure_directory,
    load_student_model,
    predict_probabilities,
    save_json,
)


def parse_named_paths(items: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH format, got: {item}")
        name, path = item.split("=", 1)
        parsed[name] = path
    return parsed


def parse_alpha_items(items: List[str]) -> List[Tuple[str, float]]:
    parsed: List[Tuple[str, float]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=ALPHA format, got: {item}")
        name, value = item.split("=", 1)
        parsed.append((name, float(value)))
    return parsed


def discover_default_checkpoints() -> Dict[str, str]:
    checkpoint_dir = "checkpoints"
    expected = {
        "fusion_alpha070": os.path.join(checkpoint_dir, "student_distill_best_alpha070.pt"),
        "fusion_alpha073": os.path.join(checkpoint_dir, "student_distill_best_alpha073.pt"),
        "fusion_alpha075": os.path.join(checkpoint_dir, "student_distill_best_alpha075.pt"),
    }
    return {name: path for name, path in expected.items() if os.path.exists(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline, teacher, student, and ensemble experiments.")
    parser.add_argument("--dataset", default="data/test.json", help="Evaluation dataset JSON.")
    parser.add_argument(
        "--student-checkpoint",
        action="append",
        default=[],
        help="Student checkpoint in NAME=PATH format. Can be repeated.",
    )
    parser.add_argument(
        "--teacher-score-file",
        help="Teacher score JSONL aligned to the evaluation dataset. Enables CLIP/InternVL/fusion baselines.",
    )
    parser.add_argument(
        "--fusion-alpha",
        action="append",
        default=[],
        help="Teacher fusion baseline in NAME=ALPHA format. Can be repeated.",
    )
    parser.add_argument("--output-dir", default="evaluation_outputs/compare_experiments")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG["threshold"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_CONFIG["max_seq_len"])
    parser.add_argument("--image-size", type=int, default=DEFAULT_CONFIG["image_size"])
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    return parser.parse_args()


def write_summary_csv(summary_rows: List[dict], output_path: str) -> None:
    fieldnames = [
        "experiment",
        "type",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc",
        "threshold",
        "positive_count",
        "negative_count",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def evaluate_student_experiments(args: argparse.Namespace, output_dir: str) -> Dict[str, dict]:
    student_paths = parse_named_paths(args.student_checkpoint) if args.student_checkpoint else discover_default_checkpoints()
    if not student_paths:
        return {}

    _, loader = create_dataloader(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: Dict[str, dict] = {}
    ensemble_probabilities: List[np.ndarray] = []
    ensemble_labels: List[int] = []
    ensemble_rows = None

    for name, checkpoint_path in student_paths.items():
        model = load_student_model(checkpoint_path, device=device)
        labels, probabilities, rows = predict_probabilities(model, loader, device=device)
        metrics = compute_metrics(labels, probabilities, threshold=args.threshold)
        results[name] = {
            "type": "student",
            "checkpoint": checkpoint_path,
            "metrics": metrics.to_dict(),
            "roc_curve": compute_roc_points(labels, probabilities),
            "precision_recall_curve": compute_precision_recall_points(labels, probabilities),
            "predictions": rows,
        }
        ensemble_probabilities.append(np.asarray(probabilities, dtype=np.float32))
        if not ensemble_labels:
            ensemble_labels = labels
            ensemble_rows = rows

    if len(ensemble_probabilities) >= 2:
        mean_probabilities = np.mean(np.stack(ensemble_probabilities, axis=0), axis=0)
        metrics = compute_metrics(ensemble_labels, mean_probabilities.tolist(), threshold=args.threshold)
        predictions = []
        for row, probability in zip(ensemble_rows, mean_probabilities.tolist()):
            enriched = dict(row)
            enriched["probability"] = float(probability)
            predictions.append(enriched)
        results["student_ensemble"] = {
            "type": "ensemble",
            "checkpoint": None,
            "metrics": metrics.to_dict(),
            "roc_curve": compute_roc_points(ensemble_labels, mean_probabilities.tolist()),
            "precision_recall_curve": compute_precision_recall_points(
                ensemble_labels, mean_probabilities.tolist()
            ),
            "predictions": predictions,
        }

    return results


def evaluate_teacher_experiments(args: argparse.Namespace, output_dir: str) -> Dict[str, dict]:
    if not args.teacher_score_file:
        return {}

    samples, _ = create_dataloader(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    results: Dict[str, dict] = {}
    for name, source in [("clip_teacher", "clip"), ("internvl_teacher", "internvl")]:
        metrics, rows = compute_teacher_baseline(
            samples=samples,
            score_file=args.teacher_score_file,
            source=source,
            threshold=args.threshold,
        )
        labels = [row["label"] for row in rows]
        probabilities = [row["probability"] for row in rows]
        results[name] = {
            "type": "teacher",
            "score_file": args.teacher_score_file,
            "metrics": metrics.to_dict(),
            "roc_curve": compute_roc_points(labels, probabilities),
            "precision_recall_curve": compute_precision_recall_points(labels, probabilities),
            "predictions": rows,
        }

    for name, alpha in parse_alpha_items(args.fusion_alpha):
        metrics, rows = compute_teacher_baseline(
            samples=samples,
            score_file=args.teacher_score_file,
            source="fusion",
            alpha=alpha,
            threshold=args.threshold,
        )
        labels = [row["label"] for row in rows]
        probabilities = [row["probability"] for row in rows]
        results[name] = {
            "type": "teacher_fusion",
            "score_file": args.teacher_score_file,
            "alpha": alpha,
            "metrics": metrics.to_dict(),
            "roc_curve": compute_roc_points(labels, probabilities),
            "precision_recall_curve": compute_precision_recall_points(labels, probabilities),
            "predictions": rows,
        }

    return results


def main() -> None:
    args = parse_args()
    ensure_directory(args.output_dir)

    student_results = evaluate_student_experiments(args, args.output_dir)
    teacher_results = evaluate_teacher_experiments(args, args.output_dir)
    combined = {**teacher_results, **student_results}

    summary_rows = []
    for name, payload in combined.items():
        metrics = payload["metrics"]
        summary_rows.append(
            {
                "experiment": name,
                "type": payload["type"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "auc": metrics["auc"],
                "threshold": metrics["threshold"],
                "positive_count": metrics["positive_count"],
                "negative_count": metrics["negative_count"],
            }
        )

    summary_rows.sort(key=lambda row: row["f1_score"], reverse=True)
    summary_json = {
        "dataset": args.dataset,
        "threshold": args.threshold,
        "experiments": combined,
        "ranking": summary_rows,
    }

    save_json(summary_json, os.path.join(args.output_dir, "comparison_summary.json"))
    write_summary_csv(summary_rows, os.path.join(args.output_dir, "comparison_summary.csv"))

    print(f"Evaluated {len(summary_rows)} experiments on {args.dataset}")
    for row in summary_rows:
        print(
            f"{row['experiment']}: "
            f"acc={row['accuracy']:.4f}, "
            f"f1={row['f1_score']:.4f}, "
            f"auc={row['auc'] if row['auc'] is not None else 'N/A'}"
        )
    print(f"Saved summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
