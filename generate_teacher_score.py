import json
from PIL import Image
import requests
import base64
import os
from tqdm import tqdm
import re
import torch
from sentence_transformers import SentenceTransformer, util

# ================= Configuration Area =================
INPUT_JSON_PATH = "data/val.json"

# Output JSONL file (single unified format)
OUT_PATH = "output/val_scores.jsonl"

CLIP_MODEL_ID = "sentence-transformers/clip-ViT-B-32"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "blaifa/InternVL3_5:8b"

# Reuse TCP connections to Ollama across requests
ollama_session = requests.Session()

# ================= Initialize CLIP Model =================
print("Loading CLIP model...")
clip_model = SentenceTransformer(CLIP_MODEL_ID)
print("CLIP model loaded successfully!")

# ================= Core Feature Extraction Functions =================

def get_clip_scores(image_path: str, texts: list[str]) -> list[dict] | None:
    """Computes per-pair CLIP cosine similarity, softmax probability, and rank."""
    try:
        image = Image.open(image_path)
        img_emb = clip_model.encode(image, convert_to_tensor=True)
        image.close()
        text_embs = clip_model.encode(texts, convert_to_tensor=True)
        cosines = util.cos_sim(img_emb, text_embs).squeeze(0)

        # Per-text softmax probability over all candidates for this image.
        # Student training uses multi-positive contrastive loss:
        # L = -log sum_{i in P} softmax(s_i) to handle varying group sizes.
        logit_scale = 100.0
        probs = torch.softmax(cosines * logit_scale, dim=0)

        # Rank by descending cosine (1 = highest)
        ranks = cosines.argsort(descending=True).argsort() + 1

        return [
            {"cosine": round(float(cosines[i]), 4),
             "prob": round(float(probs[i]), 4),
             "rank": int(ranks[i])}
            for i in range(len(texts))
        ]
    except Exception as e:
        print(f"[CLIP Error] Processing image {image_path}: {e}")
        return None

def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image to Base64 string for Ollama API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def build_sample_id(image_path: str, text: str) -> str:
    """Builds a stable sample identifier for checkpointing."""
    return f"{image_path}::{text}"

def flatten_input_records(data: list[dict]) -> list[dict]:
    """Converts the nested input structure into per-text samples.

    Supports two input formats:
      - New: {id, image: {path}, positive: [str], negative: [str]}
      - Legacy: {image_path, texts: [{text, label}]} or {image_path, text, label}
    
    Preserves the original 'id' from input data for output.
    """
    samples = []

    for item in data:
        # Preserve original ID from input
        original_id = item.get("id")
        
        # New format: image object with positive/negative text arrays
        image_obj = item.get("image")
        if isinstance(image_obj, dict):
            image_path = image_obj.get("path")
            if not image_path:
                continue
            for text in item.get("positive", []):
                if text:
                    samples.append({"image_path": image_path, "text": text, "label": 1, "original_id": original_id})
            for text in item.get("negative", []):
                if text:
                    samples.append({"image_path": image_path, "text": text, "label": 0, "original_id": original_id})
            continue

        # Legacy format
        image_path = item.get("image_path")
        if not image_path:
            continue

        texts = item.get("texts")
        if isinstance(texts, list):
            for text_item in texts:
                if not isinstance(text_item, dict):
                    continue

                text = text_item.get("text")
                if not text:
                    continue

                samples.append({
                    "image_path": image_path,
                    "text": text,
                    "label": text_item.get("label"),
                    "original_id": original_id
                })
            continue

        text = item.get("text")
        if text:
            samples.append({
                "image_path": image_path,
                "text": text,
                "label": item.get("label"),
                "original_id": original_id
            })

    return samples

def group_samples_by_image(samples: list[dict]) -> list[dict]:
    """Groups flattened samples by image while preserving input order."""
    groups: dict[str, list[dict]] = {}
    for sample in samples:
        image_path = sample.get("image_path")
        if image_path:
            groups.setdefault(image_path, []).append(sample)
    return [{"image_path": k, "samples": v} for k, v in groups.items()]

VALID_ERROR_TYPES = {"subject_mismatch", "attribute_mismatch", "spatial_mismatch", "scene_mismatch", "semantic_mismatch"}


def _normalize_score(raw: int) -> float:
    """Clamp a 0-10 integer score and normalize to [0, 1]."""
    return round(min(10, max(0, raw)) / 10.0, 2)


def _parse_internvl_fields(parsed_json: dict) -> dict | None:
    """Extract and validate score, reason, and error_type from parsed JSON."""
    reason = str(parsed_json.get("reason",
                 parsed_json.get("rationale",
                 parsed_json.get("REASON",
                 parsed_json.get("RATIONALE", ""))))).strip()
    if not reason:
        reason = "Reason not provided"

    raw_score = parsed_json.get("score", parsed_json.get("SCORE"))
    score = None
    if isinstance(raw_score, (int, float)):
        score = int(raw_score)
    elif isinstance(raw_score, str):
        m = re.search(r'\d+', raw_score)
        if m:
            score = int(m.group(0))
    if score is None:
        return None

    raw_error = parsed_json.get("error_type", parsed_json.get("ERROR_TYPE"))
    error_type = None
    if isinstance(raw_error, str):
        normalized = raw_error.strip().strip('"').lower()
        if normalized in VALID_ERROR_TYPES:
            error_type = normalized

    return {
        "score": _normalize_score(score),
        "error_type": error_type,
        "reason": reason
    }


def get_internvl_scores(base64_image: str, image_path: str, text: str) -> dict | None:
    """
    Queries InternVL for a structured alignment evaluation.
    Returns a dict with 'score' (float 0-1), 'error_type' (str|None), and 'reason' (str),
    or None on total failure.
    """
    try:
        error_type_list = ', '.join(f'"{t}"' for t in sorted(VALID_ERROR_TYPES))
        prompt = (
            f"You are an image-text alignment evaluator.\n"
            f"Text: '{text}'\n"
            f"Compare image and text, check whether there is mismatch on subject presence, attributes/actions, and spatial relations.\n"
            f"Return ONLY valid JSON with exactly three keys and no extra text:\n"
            f"{{\n"
            f"  \"reason\": \"10-20 words explaining the match or mismatch\",\n"
            f"  \"score\": a pure match score from 0 to 10,\n"
            f"  \"error_type\": null if matched, or one of {error_type_list}\n"
            f"}}\n"
            f"Do not output chain-of-thought, analysis tags, or markdown."
        )

        payload = {
            "model": OLLAMA_MODEL_NAME,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": 0.0,
            }
        }

        response = ollama_session.post(OLLAMA_API_URL, json=payload, timeout=240)
        if not response.ok:
            print(f"[Ollama HTTP Error] status={response.status_code}, body={response.text[:500]}")
            response.raise_for_status()

        result_text = response.json().get("response", "")

        try:
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(0))
                result = _parse_internvl_fields(parsed_json)
                if result is not None:
                    return result

            score_patterns = [
                r'\"score\"\s*:\s*(\d+)',
                r'\bscore\b\s*[:=]\s*(\d+)',
            ]
            for pattern in score_patterns:
                score_match = re.search(pattern, result_text, re.IGNORECASE)
                if score_match:
                    score = int(score_match.group(1))
                    short_reason = result_text.replace('\n', ' ').strip()[:180]
                    return {
                        "score": _normalize_score(score),
                        "error_type": None,
                        "reason": short_reason if short_reason else "Failed to parse reason."
                    }

            print(f"[Parse Warning] Score missing in model output: {result_text[:500]}")
            return None
        except Exception as parse_e:
            print(f"[JSON Parse Error] Model output not compliant: {result_text[:500]}. Error: {parse_e}")
            return None

    except Exception as e:
        print(f"\n[Ollama Error] Processing image {image_path}: {e}")
        return None

# ================= Helper Function: Resume from Checkpoint =================

def load_processed_ids(filepath: str) -> set:
    """Loads processed sample identifiers for checkpointing."""
    processed = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    image_path = record.get("image_path")
                    text = record.get("text")
                    if image_path and text:
                        processed.add(build_sample_id(image_path, text))
                except Exception:
                    pass
    return processed

# ================= Main Processing Flow =================

def main():
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"Input file not found: {INPUT_JSON_PATH}")
        return

    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = flatten_input_records(data)
    grouped_samples = group_samples_by_image(samples)

    processed_samples = load_processed_ids(OUT_PATH)

    if processed_samples:
        print(f"Resuming from checkpoint, skipping {len(processed_samples)} already processed items.")

    remaining = len(samples) - len(processed_samples)
    print(f"Starting processing, {remaining} items remaining...")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(OUT_PATH, 'a', encoding='utf-8') as f_out:

        for group in tqdm(grouped_samples, desc="Generating Teacher Labels"):
            image_path = group.get("image_path")
            group_items = group.get("samples", [])

            if not image_path or not group_items:
                continue

            # Skip fully-processed groups to avoid wasted CLIP/IO work on resume
            pending = [s for s in group_items
                       if s.get("text") and build_sample_id(image_path, s["text"]) not in processed_samples]
            if not pending:
                continue

            texts = [s.get("text") for s in group_items]
            clip_results = get_clip_scores(image_path, texts)

            if clip_results is None:
                print(f"Skipping {image_path} due to CLIP processing failure.")
                continue

            # Encode image once per group for all InternVL calls
            base64_image = encode_image_to_base64(image_path)

            for index, item in enumerate(group_items):
                text = item.get("text")
                label = item.get("label")
                original_id = item.get("original_id")
                sample_id = build_sample_id(image_path, text)

                if not text or sample_id in processed_samples:
                    continue

                internvl_result = get_internvl_scores(base64_image, image_path, text)

                if internvl_result is None:
                    internvl_result = {"score": None, "error_type": None, "reason": None}

                record = {
                    "id": original_id,
                    "image_path": image_path,
                    "text": text,
                    "clip": clip_results[index],
                    "internvl": internvl_result,
                    "ground_truth": {
                        "label": int(label) if label is not None else None
                    }
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                f_out.flush()

                processed_samples.add(sample_id)

    print("\nAll teacher model scores and rationales extracted successfully!")
    print(f"- Output: {OUT_PATH}")

if __name__ == "__main__":
    main()