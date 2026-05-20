git clone https://github.com/Joycejinnn/DDA4210.git
cd DDA4210
git checkout teacher_scoring_and_student_distillation

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets accelerate peft sentence-transformers openai-clip gradio scikit-learn matplotlib pandas tqdm requests pillow

## 2. Install Ollama and download InternVL3-8B
## 3. prepare data
## 4. Generate soft lables
python generate_teacher_score.py
## 5. train student model
## 6. evaluation
## 7. final madel
