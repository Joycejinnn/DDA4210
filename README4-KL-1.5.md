# KL Divergence Distillation Training Results (T=1.5)

## Training Objective

Train the student model using KL divergence loss with a temperature parameter (T=1.5) to compare the effect of different temperatures on distillation performance.

## Training Configuration

| Parameter          | Value                                              |
| ------------------ | -------------------------------------------------- |
| Fusion Weight α    | 0.7 (InternVL3)                                    |
| Loss Function      | KL Divergence + Temperature 1.5                    |
| Training Epochs    | 20 (best performance actually achieved at Epoch 2) |
| Training Samples   | 5022                                               |
| Validation Samples | 558                                                |
| Optimizer          | AdamW (lr=2e-5, weight_decay=0.01)                 |
| Batch Size         | 32                                                 |
| Student Model      | DistilBERT + 4-layer CNN                           |

## Training Results

| Epoch | Train Loss           | Val Loss             | Best Model          |
| ----- | -------------------- | -------------------- | ------------------- |
| 1     | 0.0921               | 0.0455               | ✓ (val_loss=0.0455) |
| 2     | 0.0487               | 0.0446               | ✓ (val_loss=0.0446) |
| 3     | 0.0449               | 0.0455               | -                   |
| 4     | 0.0404               | 0.0466               | -                   |
| 5~20  | Continues decreasing | Continues increasing | -                   |

**Best Model**: Epoch 2 with validation loss **0.0446** (KL divergence loss value).

## Comparison with the T=2.0 Version

| Temperature | Best Validation Loss | Model File                                            |
| ----------- | -------------------- | ----------------------------------------------------- |
| T=1.5       | 0.0446               | `checkpoints_kl_T1.5/student_distill_kl_T1.5_best.pt` |
| T=2.0       | 0.0395               | `checkpoints_kl/student_distill_kl_best.pt`           |

> **Conclusion**: T=2.0 performs slightly better than T=1.5 on this task (lower loss). It is recommended to prioritize T=2.0 during evaluation, while using T=1.5 as a comparison baseline.

## Model Usage

### Loading the T=1.5 Model

```python id="4mp2q9"
import torch
from train_student_distill_kl import StudentModel   # Outputs logits

model = StudentModel()
model.load_state_dict(torch.load("checkpoints_kl_T1.5/student_distill_kl_T1.5_best.pt", map_location="cpu"))
model.eval()
```

### Obtaining Matching Probability During Inference

```python id="8i9f0m"
def predict_prob(model, tokenizer, image_path, text, device="cpu"):
    # ... preprocessing ...
    with torch.no_grad():
        logit = model(input_ids, attention_mask, image_tensor)
        prob = torch.sigmoid(logit).item()
    return prob
```

## File Locations

* T=1.5 model weights: `checkpoints_kl_T1.5/student_distill_kl_T1.5_best.pt`
* Training script: `train_student_distill_kl_T1.5.py`
* T=2.0 model: `checkpoints_kl/student_distill_kl_best.pt`
* MSE model: `checkpoints/student_distill_best.pt`

## Conclusion

Distillation training with temperature T=1.5 was successfully completed, and the validation loss achieved its best value of 0.0446 at Epoch 2. Compared with T=2.0 (0.0395), the performance is slightly worse. Therefore, it is recommended to prioritize the T=2.0 version for subsequent evaluation and demo integration.
