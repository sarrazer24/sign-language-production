"""
download_weights.py — Download model_best.pt from Hugging Face Hub
Run once before starting the API:
    python download_weights.py
"""

import os
from huggingface_hub import hf_hub_download

# ── EDIT THESE ────────────────────────────────────────────────────────────────
REPO_ID   = "tawes/SignLanguageTextToPoseModel"
FILENAME  = "model_best.pt"
LOCAL_DIR = "weights"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(LOCAL_DIR, exist_ok=True)
path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=LOCAL_DIR)
print(f"✅ Weights saved to: {path}")
