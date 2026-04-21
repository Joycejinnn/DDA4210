# Evaluation Utilities

This module provides comprehensive utilities for computing evaluation metrics and managing results.

## Metrics Computation

### Accuracy
```python
def accuracy(y_true, y_pred):
    correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    return correct / len(y_true)
```

### Precision
```python
def precision(y_true, y_pred):
    true_positives = sum((y_t == 1) and (y_p == 1) for y_t, y_p in zip(y_true, y_pred))
    predicted_positives = sum(y_p == 1 for y_p in y_pred)
    return true_positives / predicted_positives if predicted_positives > 0 else 0
```

### Recall
```python
def recall(y_true, y_pred):
    true_positives = sum((y_t == 1) and (y_p == 1) for y_t, y_p in zip(y_true, y_pred))
    actual_positives = sum(y_t == 1 for y_t in y_true)
    return true_positives / actual_positives if actual_positives > 0 else 0
```

### F1 Score
```python
def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
```

## Results Management

### Save Results
```python
def save_results(results, file_path):
    import json
    with open(file_path, 'w') as f:
        json.dump(results, f)
```

### Load Results
```python
def load_results(file_path):
    import json
    with open(file_path, 'r') as f:
        return json.load(f)
```
