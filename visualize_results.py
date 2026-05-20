import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize experiment comparison results.")
    parser.add_argument(
        "--summary-json",
        default="evaluation_outputs/compare_experiments/comparison_summary.json",
        help="Path to comparison_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_outputs/compare_experiments/figures",
        help="Directory for generated figures.",
    )
    return parser.parse_args()


def load_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_metric_bars(summary: dict, output_dir: str) -> None:
    ranking = summary.get("ranking", [])
    if not ranking:
        return

    names = [row["experiment"] for row in ranking]
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(names))
    width = 0.18

    plt.figure(figsize=(12, 6))
    for idx, metric in enumerate(metrics):
        values = [row[metric] for row in ranking]
        plt.bar(x + idx * width, values, width=width, label=metric)

    plt.xticks(x + width * 1.5, names, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Experiment Comparison Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_bars.png"), dpi=200)
    plt.close()


def plot_topline_ranking(summary: dict, output_dir: str) -> None:
    ranking = summary.get("ranking", [])
    if not ranking:
        return

    names = [row["experiment"] for row in ranking]
    f1_scores = [row["f1_score"] for row in ranking]
    auc_scores = [0.0 if row["auc"] is None else row["auc"] for row in ranking]
    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, f1_scores, width=width, label="f1_score", color="#4C78A8")
    plt.bar(x + width / 2, auc_scores, width=width, label="auc", color="#F58518")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Topline Ranking by F1 and AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topline_ranking.png"), dpi=200)
    plt.close()


def plot_roc_curves(summary: dict, output_dir: str) -> None:
    experiments = summary.get("experiments", {})
    plt.figure(figsize=(7, 7))
    drew_curve = False
    for name, payload in experiments.items():
        roc_curve = payload.get("roc_curve")
        auc = payload.get("metrics", {}).get("auc")
        if not roc_curve:
            continue
        plt.plot(roc_curve["fpr"], roc_curve["tpr"], label=f"{name} (AUC={auc:.3f})")
        drew_curve = True

    if not drew_curve:
        plt.close()
        return

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=200)
    plt.close()


def plot_precision_recall_curves(summary: dict, output_dir: str) -> None:
    experiments = summary.get("experiments", {})
    plt.figure(figsize=(7, 7))
    drew_curve = False
    for name, payload in experiments.items():
        pr_curve = payload.get("precision_recall_curve")
        if not pr_curve:
            continue
        plt.plot(pr_curve["recall"], pr_curve["precision"], label=name)
        drew_curve = True

    if not drew_curve:
        plt.close()
        return

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=200)
    plt.close()


def plot_confusion_matrices(summary: dict, output_dir: str) -> None:
    experiments = summary.get("experiments", {})
    if not experiments:
        return

    items = list(experiments.items())
    cols = 2
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    vmax = 1
    for _, payload in items:
        matrix = payload.get("metrics", {}).get("confusion_matrix")
        if matrix:
            vmax = max(vmax, np.max(matrix))

    norm = Normalize(vmin=0, vmax=vmax)
    for idx, (name, payload) in enumerate(items):
        ax = axes[idx // cols, idx % cols]
        matrix = np.array(payload.get("metrics", {}).get("confusion_matrix", [[0, 0], [0, 0]]))
        im = ax.imshow(matrix, cmap="Blues", norm=norm)
        ax.set_title(name)
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")

    total_axes = rows * cols
    for idx in range(len(items), total_axes):
        axes[idx // cols, idx % cols].axis("off")

    fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.08, wspace=0.35, hspace=0.4)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary = load_summary(args.summary_json)
    plot_metric_bars(summary, args.output_dir)
    plot_topline_ranking(summary, args.output_dir)
    plot_roc_curves(summary, args.output_dir)
    plot_precision_recall_curves(summary, args.output_dir)
    plot_confusion_matrices(summary, args.output_dir)
    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
