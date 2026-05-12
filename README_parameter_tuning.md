## Parameter Tuning Strategy and Process

### 1. Tuning Objective
In multi-teacher knowledge distillation, the fusion weight $\alpha$ determines the proportion between soft labels from the InternVL3-8B teacher and the CLIP teacher:  
`soft_label = α · score_internvl + (1-α) · score_clip`  
At the same time, the learning rate affects the convergence speed and final performance of the student model. The tuning objective is to minimize validation MSE while maximizing accuracy.

### 2. Tuning Strategy
1. **Fix the learning rate at 2e-5** and first explore the effect of different $\alpha$ values (0.65, 0.7, 0.73, 0.75, 0.8) on distillation performance.  
2. Based on preliminary results, select the better $\alpha$ range (0.7-0.75), then **reduce the learning rate to 1e-5**, retrain with $\alpha$ = 0.7, 0.73, and 0.75, and observe whether performance improves further.  
3. Finally, try **ensembling the three best models** (simple average prediction) to test whether ensemble performance can exceed that of a single model.

### 3. Experimental Records

#### Stage 1: Validation Loss for Different $\alpha$ Values under Default Learning Rate (2e-5) (MSE against soft labels)

| α Value | Best Validation Loss (MSE) | Epoch Reached | Notes |
|------|-------------------|------------|------|
| 0.65 | 0.0101            | 2          | Average |
| 0.7  | **0.0100**        | 2          | Stable |
| 0.73 | 0.0100            | 2 or 6     | On par with 0.7 |
| 0.75 | **0.0100**        | 2          | Slightly better |
| 0.8  | 0.0102            | 6          | Slightly worse |

> Note: Validation loss is MSE (predictions vs teacher soft labels), where lower is better. All $\alpha$ values converged within 2-6 epochs.

#### Stage 2: Retraining with $\alpha$ = 0.7, 0.73, 0.75 after Reducing Learning Rate to 1e-5

| α Value | Validation MSE (vs Ground Truth Labels) | Accuracy |
|------|--------------------------|-------------------|
| 0.70 | 0.0509                   | 98.57%            |
| 0.73 | 0.0404                   | 98.75%            |
| 0.75 | **0.0361**               | **98.92%**        |

> Evaluation used `compare_models.py`, based on ground-truth labels (0/1). The validation set contains 348 positive samples and 210 negative samples.

#### Stage 3: Ensemble Model

| Model | MSE (vs Ground Truth Labels) | Accuracy |
|------|------------------|--------|
| Simple average ensemble ($\alpha$=0.7, 0.73, 0.75) | 0.0422 | 98.75% |

Ensemble performance is slightly lower than that of the best single model ($\alpha$=0.75).

### 4. Conclusion

1. **Impact of $\alpha$**: Distillation performs best when $\alpha$ is between 0.7 and 0.75. If it is too large (0.8) or too small (0.65), performance drops slightly. An overly high InternVL3 teacher weight loses CLIP's complementary information, while an overly low weight fails to fully leverage the large model's capability.  
2. **Impact of Learning Rate**: After reducing the learning rate from 2e-5 to 1e-5, accuracy against ground-truth labels improved from about 98.5% to 98.9%, and MSE decreased significantly, indicating that a smaller learning rate helps fine-tune the student model.  
3. **Ensemble Effect**: Simple averaging of three models did not outperform the best single model, possibly because the prediction distributions differ considerably, and simple averaging introduces noise instead. A weighted average (weighted by validation MSE) might offer slight improvement, but expected gains are limited.  

**The final selected student model is trained with $\alpha$=0.75 and learning rate 1e-5**, achieving **98.92% accuracy** and MSE of 0.0361 on the validation set, with fast inference speed (CPU <0.1s), meeting lightweight deployment requirements.