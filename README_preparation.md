# DDA4210 - Multi-Teacher Knowledge Distillation Data Preparation Pipeline

## Data Structure

This project uses a unified JSON data format that includes images, text descriptions, CLIP scores, InternVL model outputs, and label information.

**Example Data:**
```json
{
  "id": 407368,
  "image_path": "data/images/train2017/000000407368.jpg",
  "text": "Polar bear on rock near water in city zoo enclosure.",
  "clip": {
    "cosine": 0.3325,
    "prob": 0.5893,
    "rank": 1
  },
  "internvl": {
    "score": 0.9,
    "error_type": null,
    "reason": "The image shows a polar bear on rocks near water in what appears to be a zoo enclosure with modern architecture and a distant tower."
  },
  "ground_truth": {
    "label": 1
  }
}
```

**Field Descriptions:**
- `id`: Unique identifier of the data sample
- `image_path`: Relative path to the image file
- `text`: Text description corresponding to the image
- `clip`: CLIP model output
  - `cosine`: Cosine similarity (0-1)
  - `prob`: The softmax probability of this text among all texts for the same image. A higher `prob` means stronger relevance to the image. However, a low `prob` does not necessarily mean the text is inconsistent with the image.
  - `rank`: Relevance rank of this text to the image within the text group (1 = most relevant)
- `internvl`: InternVL model output
  - `score`: Model score (0-1)
  - `error_type`: Error type (`null` means no error). Possible values: "subject_mismatch", "attribute_mismatch", "spatial_mismatch", "scene_mismatch", "semantic_mismatch"
  - `reason`: Model reasoning process and analytical explanation
- `ground_truth`: Label information
  - `label`: Ground-truth label (0 or 1)

---

## Project Structure

- **`generate_teacher_score.py`**: Core pipeline script that uses two teacher models for scoring
  - **CLIP**: Extracts global semantic alignment (cosine similarity) from HuggingFace
  - **InternVL**: Extracts fine-grained logical reasoning and scoring through the local Ollama API
- **`data/`**: Data directory
  - `train.json`, `val.json`, `test.json`: Input datasets
  - `images/train2017/`: Image assets
- **`output/`**: Output results directory (JSONL format)
- **`data.json`**: Cleaned input dataset

---

## 1️⃣ Ollama Installation and Model Download Guide

### 1.1 Windows Installation

#### Method 1: Official Installer (Recommended)

1. Visit the [Ollama website](https://ollama.ai)
2. Click the "Download" button to download the Windows installer
3. Run the installer and follow the prompts to complete installation
4. After installation, Ollama will run in the background automatically

#### Method 2: Using Scoop (Command-Line Installation)

```powershell
scoop install ollama
```

### 1.2 Download the InternVL Model

After installing Ollama, open PowerShell or CMD and run the following command to download the InternVL model:

```bash
ollama pull blaifa/InternVL3:latest
```

**Notes:**
- `blaifa/InternVL3:latest` is the InternVL model
- The first download retrieves the full model file (about 10-15 GB), which may take time and requires a stable network connection
- The model is cached locally, so subsequent startups are faster

---

## 2️⃣ Local Ollama Usage Guide

### 2.1 Start the Ollama Service

#### Method 1: Via System Tray (GUI)

- After installation, Ollama runs in the system tray
- Click the tray icon to check status

#### Method 2: Start Service via Command Line

```bash
ollama serve
```

This starts the local API service at `http://localhost:11434`

### 2.2 Run the Model

```bash
ollama run blaifa/InternVL3:latest
```

This enters interactive chat mode, where you can directly type prompts:
```
>>> Please describe this image
<enter your prompt>
>>> 
```

### 2.3 API Call

In this project, Python calls Ollama through an HTTP API:

```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "blaifa/InternVL3:latest",
    "prompt": "your prompt",
        "stream": False
    }
)
result = response.json()
print(result['response'])
```

**Note:** Ensure the Ollama service is running before using the API.

---

## 3️⃣ Project Usage Guide

### 3.1 Environment Setup

#### Create a Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Git Bash:**
```bash
python -m venv .venv
source .venv/Scripts/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

If there is no `requirements.txt`, install required packages manually:
```bash
pip install torch torchvision torchaudio
pip install sentence-transformers
pip install pillow
pip install requests
pip install tqdm
```

### 3.2 Prepare Input Data

Place your data files in the `data/` directory and organize JSON files according to the format in the "Data Structure" section above.

### 3.3 Configure Parameters

Edit `generate_teacher_score.py` and update the following settings:

```python
# Input data file
INPUT_JSON_PATH = "data/val.json"

# Output result file
OUT_PATH = "output/val_scores.jsonl"

# CLIP model
CLIP_MODEL_ID = "sentence-transformers/clip-ViT-B-32"

# Ollama settings
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "blaifa/InternVL3:latest"
```

### 3.4 Start the Ollama Service

Before running the script, you **must** start the Ollama service:

```bash
ollama run blaifa/InternVL3:latest
```

Or, if it is already running in the background, ensure the service at `http://localhost:11434` is accessible.

### 3.5 Run the Scoring Pipeline

```bash
python generate_teacher_score.py
```

The script will:
1. Load the CLIP model
2. Process data sample by sample
3. Compute CLIP similarity scores
4. Call InternVL to obtain reasoning and detailed scores
5. Save results to `output/val_scores.jsonl`

### 3.6 Output Format

Output is in JSONL format (one JSON object per line):

```json
{
  "image_path": "images/train2017/image_001.jpg",
  "texts": ["description text 1", "description text 2", "description text 3"],
  "label": 0,
  "clip_scores": [
    {"cosine": 0.7234, "prob": 0.4521, "rank": 1},
    {"cosine": 0.6891, "prob": 0.3245, "rank": 2},
    {"cosine": 0.5123, "prob": 0.2234, "rank": 3}
  ],
  "internvl_response": "model reasoning process and analysis...",
  "internvl_detail_score": 0.85
}
```

---

## 🐛 Frequently Asked Questions

### Issue 1: Unable to Connect to Ollama

**Symptom:**
```
requests.exceptions.ConnectionError: Failed to establish a new connection
```

**Solution:**
1. Ensure the Ollama service is running: `ollama run blaifa/InternVL3:latest`
2. Check whether the API URL is correct (default: `http://localhost:11434`)
3. Check whether the firewall is blocking local connections

### Issue 2: Model Load Failure

**Symptom:**
```
Error: model 'blaifa/InternVL3:latest' not found
```

**Solution:**
1. Download the model: `ollama pull blaifa/InternVL3:latest`
2. Verify the model: `ollama list`

### Issue 3: Out of Memory

**Symptom:**
```
CUDA out of memory or Memory allocation failed
```

**Solution:**
- Reduce batch size (adjust in the script)
- Close other programs to free memory

### Issue 4: Network Timeout

**Symptom:**
```
requests.exceptions.ReadTimeout
```

**Solution:**
Increase the timeout in `generate_teacher_score.py`:
```python
ollama_session.request(..., timeout=300)  # 5-minute timeout
```

---

## 📋 Dependency List

| Package | Version | Purpose |
|------|------|------|
| torch | 2.0+ | Deep learning framework |
| sentence-transformers | 2.2+ | CLIP model loading |
| Pillow | 10.0+ | Image processing |
| requests | 2.31+ | HTTP requests (Ollama API) |
| tqdm | 4.65+ | Progress bar display |

---

## 📝 Usage Example

Complete workflow:

```bash
# 1. Activate the virtual environment
source .venv/Scripts/activate

# 2. Start the Ollama service (in a new terminal window)
ollama run blaifa/InternVL3:latest

# 3. Run the scoring script
python generate_teacher_score.py

# 4. Check results
cat output/val_scores.jsonl | head -n 1
```

---

## 📚 Reference Resources

- [Ollama Official Website](https://ollama.ai)
- [InternVL GitHub](https://github.com/OpenGVLab/InternVL)
- [CLIP Documentation](https://github.com/openai/CLIP)
- [Sentence Transformers](https://www.sbert.net/)
