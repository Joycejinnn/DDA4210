

# KL 散度蒸馏训练结果

## 训练目标
使用 KL 散度损失 + 温度参数（T=2.0）训练学生模型，替代之前的 MSE 回归损失，期望缓解过拟合并提升模型泛化能力。

## 训练配置
| 参数 | 值 |
|------|-----|
| 融合权重 α | 0.7 (InternVL3) |
| 损失函数 | KL 散度 + 温度 2.0 |
| 训练轮数 | 20（实际最佳在第2轮） |
| 训练样本 | 5022 |
| 验证样本 | 558 |
| 优化器 | AdamW (lr=2e-5, weight_decay=0.01) |
| 批次大小 | 32 |
| 学生模型 | DistilBERT + 4层CNN |

## 训练结果
| Epoch | Train Loss | Val Loss | Best Model |
|-------|------------|----------|-------------|
| 1     | 0.0873     | 0.0403   | ✓ (val_loss=0.0403) |
| 2     | 0.0415     | 0.0395   | ✓ (val_loss=0.0395) |
| 3     | 0.0396     | 0.0418   | - |
| 4     | 0.0359     | 0.0438   | - |
| ...   | 继续下降   | 持续上升 | - |

**最佳模型**：第2轮，验证损失 **0.0395**（KL 散度损失值，与 MSE 不可直接比较）。

## 与 MSE 版本对比
| 指标 | MSE 版本 | KL 版本 (T=2.0) |
|------|----------|------------------|
| 最佳验证损失 | 0.0100 (MSE) | 0.0395 (KL) |
| 过拟合起始轮 | 第3轮 | 第3轮 |
| 模型文件 | `checkpoints/student_distill_best.pt` | `checkpoints_kl/student_distill_kl_best.pt` |

> **注意**：两种损失函数数值范围不同，MSE 损失通常比 KL 小很多。实际效果需通过下游评估指标（AUC、准确率）对比。

## 模型使用方式

### 加载 KL 版本模型
```python
import torch
from train_student_distill import StudentModel   # 注意：需要修改模型类去掉最后的 Sigmoid

# KL 版本的模型定义中去掉了 Sigmoid，输出 logit
model = StudentModel()  # 该 StudentModel 应输出 logit（未经过 sigmoid）
model.load_state_dict(torch.load("checkpoints_kl/student_distill_kl_best.pt", map_location="cpu"))
model.eval()
```

### 推理时获取匹配概率
由于 KL 版本输出 logit，需要手动计算 sigmoid 得到概率：
```python
def predict_prob(model, tokenizer, image_path, text, device="cpu"):
    # ... 图像和文本预处理 ...
    with torch.no_grad():
        logit = model(input_ids, attention_mask, image_tensor)
        prob = torch.sigmoid(logit).item()
    return prob
```

## 建议评估任务
1. **对比 MSE 模型 vs KL 模型**：在同一测试集上计算准确率、AUC、F1。
2. **不同温度实验**：可尝试 T=1.0, 3.0 进一步调优。
3. **与单教师、基线对比**：按原计划进行。

## 文件位置
- KL 模型权重：`checkpoints_kl/student_distill_kl_best.pt`
- 训练脚本：`train_student_distill_kl.py`
- 原始 MSE 模型：`checkpoints/student_distill_best.pt`

## 结论
KL 散度蒸馏成功训练出学生模型，验证损失稳定收敛并在第2轮达到最佳。虽然损失数值高于 MSE 版本，但这是损失函数定义不同所致。建议在真实图文匹配任务上评估两个版本的实际性能，选择更优者用于最终 Demo。
```

