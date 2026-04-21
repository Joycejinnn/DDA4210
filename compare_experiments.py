# compare_experiments.py

"""
This script provides functionality to compare various experimentation approaches in the context of knowledge distillation. It evaluates:

- Baseline approach
- CLIP-only approach
- InternVL-only approach
- Fusion-based approach

"""

import numpy as np
import pandas as pd

class ExperimentComparison:
    def __init__(self, baseline_data, clip_data, internvl_data, fusion_data):
        self.baseline_data = baseline_data
        self.clip_data = clip_data
        self.internvl_data = internvl_data
        self.fusion_data = fusion_data

    def compute_metrics(self, data):
        """Compute evaluation metrics from experiment data."""
        metrics = {
            'accuracy': np.mean(data['accuracy']),
            'f1_score': np.mean(data['f1_score']),
            'precision': np.mean(data['precision']),
            'recall': np.mean(data['recall']),
        }
        return metrics

    def compare_experiments(self):
        """Compare all experiments and return a summary of metrics."""
        metrics = {
            'Baseline': self.compute_metrics(self.baseline_data),
            'CLIP': self.compute_metrics(self.clip_data),
            'InternVL': self.compute_metrics(self.internvl_data),
            'Fusion': self.compute_metrics(self.fusion_data)
        }
        return metrics

if __name__ == '__main__':
    # Sample data
    baseline_data = pd.DataFrame({'accuracy': [0.85, 0.87], 'f1_score': [0.80, 0.82], 'precision': [0.83, 0.84], 'recall': [0.79, 0.81]})
    clip_data = pd.DataFrame({'accuracy': [0.86, 0.88], 'f1_score': [0.81, 0.83], 'precision': [0.85, 0.86], 'recall': [0.80, 0.82]})
    internvl_data = pd.DataFrame({'accuracy': [0.84, 0.86], 'f1_score': [0.78, 0.80], 'precision': [0.82, 0.83], 'recall': [0.76, 0.78]})
    fusion_data = pd.DataFrame({'accuracy': [0.87, 0.89], 'f1_score': [0.83, 0.85], 'precision': [0.86, 0.87], 'recall': [0.81, 0.83]})

    experiment_comparison = ExperimentComparison(baseline_data, clip_data, internvl_data, fusion_data)
    results = experiment_comparison.compare_experiments()
    print(results)