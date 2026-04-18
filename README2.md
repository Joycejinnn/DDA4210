# 知识蒸馏训练：InternVL3 + CLIP → 轻量学生模型

## 项目背景
我们训练了一个轻量级学生模型（DistilBERT + 轻量CNN），用于图文匹配任务。教师模型为：
- **InternVL3-8B**（通过API获得分数）
- **CLIP-ViT-B/32**（本地提取相似度）

采用加权平均融合策略：  
`teacher_soft_label = 0.7 * internvl_score + 0.3 * clip_score`

学生模型通过回归（MSE损失）学习教师软标签，从而具备图文匹配能力。

## 当前产出
- 训练脚本：`train_student_distill.py`
- 最佳模型权重：`checkpoints/student_distill_best.pt`（验证损失最低的模型）
- 教师分数文件：`output/train_scores.jsonl`（包含每张图片的 clip.prob 和 internvl.score 以及真实标签）

## 模型信息
| 项目 | 说明 |
|------|------|
| 输入 | 图片（RGB） + 文本（自然语言） |
| 输出 | 匹配分数（0~1，越接近1越匹配） |
| 结构 | DistilBert文本编码器 + 4层CNN图像编码器 + 融合层 |
| 训练样本 | 5022对（训练），558对（验证） |
| 损失函数 | MSE（蒸馏损失） |
| 最佳验证损失 | 0.0100（第2轮） |
| 融合权重 | α = 0.7（InternVL3权重） |

## 数据格式
`output/train_scores.jsonl` 每行示例：
```json
{
  "image_path": "data/images/train2017/000000199602.jpg",
  "text": "A woman standing on a beach while holding a kite.",
  "clip": {"prob": 0.0536},
  "internvl": {"score": 1.0},
  "ground_truth": {"label": 1}
}
```
- `clip.prob`：CLIP 模型输出的匹配概率（0~1）
- `internvl.score`：InternVL3 评分（0~1）
- `ground_truth.label`：真实标签（1=匹配，0=不匹配）

## 如何使用学生模型

### 加载模型
```python
import torch
from train_student_distill import StudentModel

model = StudentModel()
model.load_state_dict(torch.load("checkpoints/student_distill_best.pt", map_location="cpu"))
model.eval()
```

### 推理函数示例
```python
from transformers import DistilBertTokenizer
from PIL import Image
import torchvision.transforms as transforms

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def predict(image_path, text):
    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
    enc = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
    with torch.no_grad():
        score = model(enc['input_ids'], enc['attention_mask'], image).item()
    return score
```


评估指标：
- 准确率（Accuracy）
- AUC（ROC曲线下面积）
- F1-score
- 可选项：不同阈值下的精确率/召回率

数据划分：使用 `output/train_scores.jsonl` 中的 `ground_truth.label` 作为真实标签。注意该文件已经包含训练/验证/测试样本？目前只有一份，你可以自行划分或使用原始数据集中的划分文件（如 `data/train.json`, `data/val.json`, `data/test.json`）。

## 注意事项
- 学生模型输入图片尺寸为 224×224，文本最大长度 64。
- 模型权重文件较大（~255 MB），已上传至网盘：[请同学C提供链接]
- 训练日志显示存在轻微过拟合（验证损失在第2轮后上升），可尝试增加 Dropout 或早停。

