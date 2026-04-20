以下是您提供的复现步骤的 Markdown 格式原内容演示：

```markdown
# 克隆仓库并切换到 yiyao 分支
git clone https://github.com/Joycejinnn/DDA4210.git
cd DDA4210
git checkout yiyao

# 创建虚拟环境 (Python 3.10)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 安装依赖 (CPU 版 PyTorch)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets accelerate peft sentence-transformers openai-clip gradio scikit-learn matplotlib pandas tqdm requests pillow
```

## 2. 安装 Ollama 并下载 InternVL3-8B

```bash
# 安装 Ollama (从官网下载安装包)
# 然后拉取模型
ollama pull blaifa/InternVL3_5:8b
```

## 3. 准备数据

将 COCO 2017 训练集图片放入 `data/images/train2017/`。

确保 `data/train.json` 存在（包含 `image_path`, `text`, `label` 字段）。

若缺少图片，运行下载脚本（只下载需要的图片）：

```bash
python download_images.py
```

## 4. 生成教师软标签

```bash
# 确保 Ollama 服务已启动（ollama serve）
python generate_teacher_score.py
```

成功后会生成 `output/train_scores.jsonl`。

## 5. 训练学生模型

直接训练（使用默认 α=0.75）：  
（`CONFIG["best_model_name"]` 改成对应的 `"best_model_name_alpha075"`）

```bash
python train_student_distill.py
```

若要训练不同 α 值（如 0.7, 0.73），修改脚本中的 `CONFIG["alpha"]` 和 `CONFIG["best_model_name"]`（改成对应的 `"best_model_name_alpha070"`, `"best_model_name_alpha073"` 以适配 `compare_models.py`），然后运行。

## 6. 评估模型

```bash
python compare_models.py
```

该脚本会输出各单模型及集成的 MSE 和准确率。

## 7. 使用最终模型

最佳模型为 `checkpoints/student_distill_best_alpha075.pt`。加载示例：

```python
import torch
from train_student_distill import StudentModel

model = StudentModel()
model.load_state_dict(torch.load("checkpoints/student_distill_best_alpha075.pt", map_location="cpu"))
model.eval()
```