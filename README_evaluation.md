# WZW Branch Evaluation Guide

This README is for the `wzw` branch only. It explains how to run the evaluation and visualization scripts for the model comparison part of the DDA4210 project.

## 1. What This Branch Does

The `wzw` branch is responsible for the evaluation stage:

- evaluate a single student model
- compare multiple student models
- evaluate an ensemble of student models
- optionally compare teacher-only baselines:
  - CLIP teacher
  - InternVL teacher
  - weighted fusion teacher
- generate report-ready figures:
  - metric bar chart
  - F1/AUC ranking chart
  - ROC curves
  - Precision-Recall curves
  - confusion matrix heatmaps

## 2. Expected Inputs

### Required

- dataset JSON:
  - `data/val.json` or
  - `data/test.json`

- student checkpoints:
  - `.pt` files produced by the training branch

Default checkpoint names recognized automatically:

```text
checkpoints/student_distill_best_alpha070.pt
checkpoints/student_distill_best_alpha073.pt
checkpoints/student_distill_best_alpha075.pt
```

### Optional

- teacher score JSONL for the same evaluation dataset

Example:

```text
output/test_scores.jsonl
```

This file is needed if you want to compare:

- `clip_teacher`
- `internvl_teacher`
- teacher fusion baselines such as `alpha=0.70`, `0.73`, `0.75`

Important:
The teacher score file must be aligned to the same dataset you are evaluating. If you run on `data/test.json`, the JSONL should contain scores for the flattened samples from `data/test.json`, not from `train.json`.

### DistilBERT local files

The student checkpoints depend on `distilbert-base-uncased`. If your machine cannot download from Hugging Face, place a local copy of that model directory in the project and pass it through `--text-model-name`.

The local directory should contain files such as:

```text
config.json
tokenizer.json
tokenizer_config.json
vocab.txt
special_tokens_map.json
model.safetensors or pytorch_model.bin
```

## 3. Install Dependencies

Use the same environment as the training branch. At minimum you need:

```bash
pip install torch torchvision transformers scikit-learn matplotlib numpy pillow
```

## 4. Evaluate One Student Model

Example:

```bash
python evaluate_student_model.py --checkpoint checkpoints/student_distill_best_alpha075.pt --dataset data/test.json
```

Output:

```text
evaluation_outputs/single_model/single_model_results.json
```

This file includes:

- metrics
- ROC points
- Precision-Recall points
- per-sample predictions

## 5. Compare Multiple Student Models

### Option A: use default checkpoint names

If the three default checkpoint files exist, just run:

```bash
python compare_experiments.py --dataset data/test.json
```

### Option B: specify checkpoint names manually

```bash
python compare_experiments.py ^
  --dataset data/test.json ^
  --student-checkpoint alpha070=checkpoints/student_distill_best_alpha070.pt ^
  --student-checkpoint alpha073=checkpoints/student_distill_best_alpha073.pt ^
  --student-checkpoint alpha075=checkpoints/student_distill_best_alpha075.pt
```

Output:

```text
evaluation_outputs/compare_experiments/comparison_summary.json
evaluation_outputs/compare_experiments/comparison_summary.csv
```

The script also creates a `student_ensemble` result automatically if at least two student checkpoints are provided.

## 6. Compare Student Models with Teacher Baselines

If you also have a teacher score JSONL for the same dataset, run:

```bash
python compare_experiments.py ^
  --dataset data/test.json ^
  --teacher-score-file output/test_scores.jsonl ^
  --fusion-alpha fusion070=0.70 ^
  --fusion-alpha fusion073=0.73 ^
  --fusion-alpha fusion075=0.75 ^
  --student-checkpoint alpha070=checkpoints/student_distill_best_alpha070.pt ^
  --student-checkpoint alpha073=checkpoints/student_distill_best_alpha073.pt ^
  --student-checkpoint alpha075=checkpoints/student_distill_best_alpha075.pt
```

If you need to use a local DistilBERT directory:

```bash
python compare_experiments.py ^
  --dataset data/test.json ^
  --text-model-name local_models/distilbert-base-uncased ^
  --student-checkpoint alpha070=checkpoints/student_distill_best_alpha070.pt ^
  --student-checkpoint alpha073=checkpoints/student_distill_best_alpha073.pt ^
  --student-checkpoint alpha075=checkpoints/student_distill_best_alpha075.pt
```

This will compare:

- `clip_teacher`
- `internvl_teacher`
- the specified fusion teacher baselines
- each student checkpoint
- `student_ensemble`

## 7. Visualize Results

After generating `comparison_summary.json`, run:

```bash
python visualize_results.py
```

Or specify the summary file manually:

```bash
python visualize_results.py --summary-json evaluation_outputs/compare_experiments/comparison_summary.json
```

Generated figures:

```text
evaluation_outputs/compare_experiments/figures/metric_bars.png
evaluation_outputs/compare_experiments/figures/topline_ranking.png
evaluation_outputs/compare_experiments/figures/roc_curves.png
evaluation_outputs/compare_experiments/figures/precision_recall_curves.png
evaluation_outputs/compare_experiments/figures/confusion_matrices.png
```

## 8. Evaluate Only the Ensemble

If you want to run ensemble evaluation separately:

```bash
python ensemble_evaluation.py ^
  --dataset data/test.json ^
  --checkpoint checkpoints/student_distill_best_alpha070.pt ^
  --checkpoint checkpoints/student_distill_best_alpha073.pt ^
  --checkpoint checkpoints/student_distill_best_alpha075.pt
```

Output:

```text
evaluation_outputs/ensemble/ensemble_results.json
```

## 9. Recommended Workflow for Student E

Use this order:

1. collect the final student checkpoints from the training branch
2. put them into `checkpoints/`
3. prepare teacher scores for `data/test.json` if teacher comparison is needed
4. run `compare_experiments.py`
5. run `visualize_results.py`
6. use `comparison_summary.csv` and the figures in the final report or slides

## 10. Notes

- `wzw` branch now expects real checkpoints and real evaluation inputs. The old placeholder metrics are no longer used.
- if AUC is shown as `N/A`, it usually means the evaluated labels contain only one class after filtering
- if teacher baselines are missing, student-only comparison can still run normally
- if checkpoint names differ from the default names, use `--student-checkpoint NAME=PATH`
