# DDA4210 - 多教师知识蒸馏数据准备管道

这是一个多教师知识蒸馏（Knowledge Distillation）数据准备管道，用于生成高质量的"软标签"和推理过程，以训练一个较小的学生模型（如Qwen2.5），特别优化用于仅CPU环境。

## 项目结构

- **`generate_teacher_score.py`**：核心管道脚本，使用两个教师模型进行评分
  - **CLIP**：从HuggingFace提取全局语义对齐（余弦相似度）
  - **InternVL**：通过本地Ollama API提取细粒度逻辑推理和评分
- **`data/`**：数据目录
  - `train.json`, `val.json`, `test.json`：输入数据集
  - `images/train2017/`：图像资源
- **`output/`**：输出结果目录（JSONL格式）
- **`data.json`**：清理后的输入数据集

---

## 1️⃣ Ollama 安装和下载教程

### 1.1 Windows 安装

#### 方法一：使用官方安装程序（推荐）

1. 访问 [Ollama官网](https://ollama.ai)
2. 点击 "Download" 按钮下载 Windows 安装程序
3. 运行安装程序，按照提示完成安装
4. 安装完成后，Ollama会自动在后台运行

#### 方法二：使用 Scoop（命令行安装）

```powershell
scoop install ollama
```

### 1.2 下载 InternVL 模型

安装完Ollama后，打开PowerShell或CMD，运行以下命令下载InternVL模型：

```bash
ollama pull blaifa/InternVL3_5:8b
```

**说明：**
- `blaifa/InternVL3_5:8b` 是8B参数的InternVL模型
- 首次下载时会下载完整模型文件（约10-15GB），需要较长时间和网络连接
- 模型会缓存在本地，后续启动会更快

### 1.3 验证安装

```bash
ollama list
```

如果看到类似输出说明安装成功：
```
NAME                    ID              SIZE      MODIFIED
blaifa/InternVL3_5:8b   abc123...       10.2GB    2 minutes ago
```

---

## 2️⃣ Ollama 本地使用教程

### 2.1 启动 Ollama 服务

#### 方法一：通过系统托盘（图形界面）

- Ollama安装后会在系统托盘运行
- 点击托盘图标可以查看状态

#### 方法二：通过命令行启动服务

```bash
ollama serve
```

这会在 `http://localhost:11434` 启动本地API服务

### 2.2 运行模型

```bash
ollama run blaifa/InternVL3_5:8b
```

进入交互式对话模式，你可以直接输入问题：
```
>>> 请描述这张图片
<输入提示词>
>>> 
```

### 2.3 API 调用

项目中使用Python通过HTTP API调用Ollama：

```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "blaifa/InternVL3_5:8b",
        "prompt": "你的提示词",
        "stream": False
    }
)
result = response.json()
print(result['response'])
```

**注意：** 确保Ollama服务已启动，API才能正常使用

---

## 3️⃣ 项目使用教程

### 3.1 环境准备

#### 创建虚拟环境

**Windows PowerShell：**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Git Bash：**
```bash
python -m venv .venv
source .venv/Scripts/activate
```

#### 安装依赖

```bash
pip install -r requirements.txt
```

若无requirements.txt，手动安装必要的包：
```bash
pip install torch torchvision torchaudio
pip install sentence-transformers
pip install pillow
pip install requests
pip install tqdm
```

### 3.2 准备输入数据

在 `data/` 目录下放置你的数据文件

### 3.3 配置参数

编辑 `generate_teacher_score.py`，修改以下配置：

```python
# 输入数据文件
INPUT_JSON_PATH = "data/val.json"

# 输出结果文件
OUT_PATH = "output/val_scores.jsonl"

# CLIP模型
CLIP_MODEL_ID = "sentence-transformers/clip-ViT-B-32"

# Ollama配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "blaifa/InternVL3_5:8b"
```

### 3.4 启动 Ollama 服务

运行脚本前，**必须**启动Ollama服务：

```bash
ollama run blaifa/InternVL3_5:8b
```

或者如果已在后台运行，确保服务在 `http://localhost:11434` 可访问

### 3.5 运行评分管道

```bash
python generate_teacher_score.py
```

脚本会：
1. 加载CLIP模型
2. 逐条处理数据
3. 计算CLIP相似度分数
4. 调用InternVL获取推理过程和细节评分
5. 将结果保存到 `output/val_scores.jsonl`

### 3.6 输出格式

输出JSONL文件格式（每行一个JSON对象）：

```json
{
  "image_path": "images/train2017/image_001.jpg",
  "texts": ["描述文本1", "描述文本2", "描述文本3"],
  "label": 0,
  "clip_scores": [
    {"cosine": 0.7234, "prob": 0.4521, "rank": 1},
    {"cosine": 0.6891, "prob": 0.3245, "rank": 2},
    {"cosine": 0.5123, "prob": 0.2234, "rank": 3}
  ],
  "internvl_response": "模型的推理过程和分析...",
  "internvl_detail_score": 0.85
}
```

---

## 🐛 常见问题

### 问题1：无法连接到Ollama

**症状：**
```
requests.exceptions.ConnectionError: Failed to establish a new connection
```

**解决方案：**
1. 确保Ollama服务已启动：`ollama run blaifa/InternVL3_5:8b`
2. 检查API地址是否正确（默认：`http://localhost:11434`）
3. 检查防火墙是否阻止了本地连接

### 问题2：模型加载失败

**症状：**
```
Error: model 'blaifa/InternVL3_5:8b' not found
```

**解决方案：**
1. 下载模型：`ollama pull blaifa/InternVL3_5:8b`
2. 验证模型：`ollama list`

### 问题3：内存不足

**症状：**
```
CUDA out of memory 或 Memory allocation failed
```

**解决方案：**
- 减少批处理大小（在脚本中调整）
- 关闭其他程序释放内存

### 问题4：网络超时

**症状：**
```
requests.exceptions.ReadTimeout
```

**解决方案：**
在 `generate_teacher_score.py` 中增加超时时间：
```python
ollama_session.request(..., timeout=300)  # 5分钟超时
```

---

## 📋 依赖包列表

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | 2.0+ | 深度学习框架 |
| sentence-transformers | 2.2+ | CLIP模型加载 |
| Pillow | 10.0+ | 图像处理 |
| requests | 2.31+ | HTTP请求（Ollama API） |
| tqdm | 4.65+ | 进度条显示 |

---

## 📝 使用示例

完整的工作流程：

```bash
# 1. 激活虚拟环境
source .venv/Scripts/activate

# 2. 启动Ollama服务（新终端窗口）
ollama run blaifa/InternVL3_5:8b

# 3. 运行评分脚本
python generate_teacher_score.py

# 4. 查看结果
cat output/val_scores.jsonl | head -n 1
```

---

## 📚 参考资源

- [Ollama 官网](https://ollama.ai)
- [InternVL GitHub](https://github.com/OpenGVLab/InternVL)
- [CLIP 文档](https://github.com/openai/CLIP)
- [Sentence Transformers](https://www.sbert.net/)