
# KL 散度蒸馏训练结果（T=1.5）

## 训练目标
使用 KL 散度损失 + 温度参数（T=1.5）训练学生模型，对比不同温度对蒸馏效果的影响。

## 训练配置
| 参数 | 值 |
|------|-----|
| 融合权重 α | 0.7 (InternVL3) |
| 损失函数 | KL 散度 + 温度 1.5 |
| 训练轮数 | 20（实际最佳在第2轮） |
| 训练样本 | 5022 |
| 验证样本 | 558 |
| 优化器 | AdamW (lr=2e-5, weight_decay=0.01) |
| 批次大小 | 32 |
| 学生模型 | DistilBERT + 4层CNN |

## 训练结果
| Epoch | Train Loss | Val Loss | Best Model |
|-------|------------|----------|-------------|
| 1     | 0.0921     | 0.0455   | ✓ (val_loss=0.0455) |
| 2     | 0.0487     | 0.0446   | ✓ (val_loss=0.0446) |
| 3     | 0.0449     | 0.0455   | - |
| 4     | 0.0404     | 0.0466   | - |
| 5~20  | 继续下降   | 持续上升 | - |

**最佳模型**：第2轮，验证损失 **0.0446**（KL 散度损失值）。

## 与 T=2.0 版本对比
| 温度 | 最佳验证损失 | 模型文件 |
|------|-------------|----------|
| T=1.5 | 0.0446 | `checkpoints_kl_T1.5/student_distill_kl_T1.5_best.pt` |
| T=2.0 | 0.0395 | `checkpoints_kl/student_distill_kl_best.pt` |

> **结论**：T=2.0 在本任务中略优于 T=1.5（损失更低）。建议评估时以 T=2.0 为主，T=1.5 作为对比。

## 模型使用方式

### 加载 T=1.5 模型
```python
import torch
from train_student_distill_kl import StudentModel   # 输出 logit

model = StudentModel()
model.load_state_dict(torch.load("checkpoints_kl_T1.5/student_distill_kl_T1.5_best.pt", map_location="cpu"))
model.eval()
```

### 推理时获取匹配概率
```python
def predict_prob(model, tokenizer, image_path, text, device="cpu"):
    # ... 预处理 ...
    with torch.no_grad():
        logit = model(input_ids, attention_mask, image_tensor)
        prob = torch.sigmoid(logit).item()
    return prob
```

## 文件位置
- T=1.5 模型权重：`checkpoints_kl_T1.5/student_distill_kl_T1.5_best.pt`
- 训练脚本：`train_student_distill_kl_T1.5.py`
- T=2.0 模型：`checkpoints_kl/student_distill_kl_best.pt`
- MSE 模型：`checkpoints/student_distill_best.pt`

## 结论
温度 T=1.5 的蒸馏训练成功完成，验证损失在第2轮达到最佳 0.0446。相比 T=2.0（0.0395）性能略低，建议优先使用 T=2.0 版本进行后续评估和 Demo 集成。

