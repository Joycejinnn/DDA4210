# 📊 Training Status Summary

| Item | Value | Description |
|------|------|------|
| **Device** | CPU | Your training ran entirely on CPU (no GPU detected). |
| **Loaded Samples** | 5580 | There are 5,580 valid image-text pairs in the teacher score file. |
| **Fusion Weight α** | 0.8 | InternVL3 teacher contributes 80%, and CLIP contributes 20%. |
| **Example Soft Label** | 0.8030 | The fused teacher score of the first sample (close to 0.8). |
| **Training Set Size** | 5022 | 90% of the data is used for training. |
| **Validation Set Size** | 558 | 10% of the data is used for validation. |
| **Loaded Student Model** | DistilBERT | The text encoder is `distilbert-base-uncased`. |

# 📦 About Model Size

From this output line:
model.safetensors: 100% | 268M/268M [02:28<00:00, 1.80MB/s]

it is clear that the weight file size of the `distilbert-base-uncased` model is **268 MB**.

- This 268 MB is the saved safetensors file size, which contains all model parameters.
- The actual parameter count is about **66 million**, because DistilBERT is a lightweight version of BERT with roughly half the parameters of BERT-base.
- A disk size of 268 MB is reasonable: weights are stored in 32-bit floating point format, each parameter takes 4 bytes, so $66M \times 4B \approx 264$ MB, and with additional metadata it becomes 268 MB.

