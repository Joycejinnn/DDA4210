# KL Divergence Distillation Training Results

## Training Objective

Train the student model using KL divergence loss with a temperature parameter (T=2.0), replacing the previous MSE regression loss, with the goal of mitigating overfitting and improving model generalization.

## Training Configuration

| Parameter          | Value                              |
| ------------------ | ---------------------------------- |
| Fusion Weight α    | 0.7 (InternVL3)                    |
| Loss Function      | KL Divergence + Temperature 2.0    |
| Training Epochs    | 20 (actual best at Epoch 2)        |
| Training Samples   | 5022                               |
| Validation Samples | 558                                |
| Optimizer          | AdamW (lr=2e-5, weight_decay=0.01) |
| Batch Size         | 32                                 |
| Student Model      | DistilBERT + 4-layer CNN           |

## Training Results

| Epoch | Train Loss           | Val Loss             | Best Model          |
| ----- | -------------------- | -------------------- | ------------------- |
| 1     | 0.0873               | 0.0403               | ✓ (val_loss=0.0403) |
| 2     | 0.0415               | 0.0395               | ✓ (val_loss=0.0395) |
| 3     | 0.0396               | 0.0418               | -                   |
| 4     | 0.0359               | 0.0438               | -                   |
| ...   | Continues decreasing | Continues increasing | -                   |

**Best Model**: Epoch 2, validation loss **0.0395** (KL divergence loss value, not directly comparable with MSE).

## Comparison with the MSE Version

| Metric               | MSE Version                           | KL Version (T=2.0)                          |
| -------------------- | ------------------------------------- | ------------------------------------------- |
| Best Validation Loss | 0.0100 (MSE)                          | 0.0395 (KL)                                 |
| Overfitting Starts   | Epoch 3                               | Epoch 3                                     |
| Model File           | `checkpoints/student_distill_best.pt` | `checkpoints_kl/student_distill_kl_best.pt` |

> **Note**: The numerical scales of the two loss functions are different. MSE losses are typically much smaller than KL divergence losses. Actual performance should be compared using downstream evaluation metrics such as AUC and accuracy.

## Model Usage

### Load the KL Version Model

```python
import torch
from train_student_distill import StudentModel   # Note: modify the model class to remove the final Sigmoid

# In the KL version, the model definition removes the Sigmoid and outputs logits
model = StudentModel()  # This StudentModel should output logits (without sigmoid applied)
model.load_state_dict(torch.load("checkpoints_kl/student_distill_kl_best.pt", map_location="cpu"))
model.eval()
```

### Obtain Matching Probability During Inference

Since the KL version outputs logits, sigmoid must be applied manually to obtain probabilities:

```python
def predict_prob(model, tokenizer, image_path, text, device="cpu"):
    # ... image and text preprocessing ...
    with torch.no_grad():
        logit = model(input_ids, attention_mask, image_tensor)
        prob = torch.sigmoid(logit).item()
    return prob
```

## Recommended Evaluation Tasks

1. **Compare the MSE model vs. the KL model**: compute accuracy, AUC, and F1 score on the same test set.
2. **Experiment with different temperatures**: try T=1.0 and T=3.0 for further tuning.
3. **Compare with single-teacher and baseline models**: proceed according to the original plan.

## File Locations

* KL model weights: `checkpoints_kl/student_distill_kl_best.pt`
* Training script: `train_student_distill_kl.py`
* Original MSE model: `checkpoints/student_distill_best.pt`

## Conclusion

KL divergence distillation successfully trained the student model. The validation loss converged stably and reached the best result at Epoch 2. Although the loss value is higher than that of the MSE version, this is due to the difference in loss function definitions. It is recommended to evaluate both versions on real image-text matching tasks and select the better-performing model for the final demo.
