from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import SignLanguageInference
import numpy as np
import os

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

weights_path = os.path.join(BASE_DIR, "weights", "model_best.pt")

stats_path = os.path.join(BASE_DIR, "data", "stats.pt")
infer = SignLanguageInference(
    weights_path=weights_path,
    stats_path=stats_path,
    tokenizer_path=os.getenv("T5_PATH", "t5-small"),
)

class GenerateRequest(BaseModel):
    text: str
    n_frames: int = 60
    guidance_scale: float = 3.0

@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    try:
        poses = infer.generate(req.text, req.n_frames, req.guidance_scale)
        # poses shape: (T, 151, 3)
        return {
            "n_frames": len(poses),
            "n_keypoints": 151,
            "poses": poses.tolist(),           # raw (T,151,3)
            "openpose": infer.generate_openpose_json(
                req.text, req.n_frames, req.guidance_scale
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ml service running ✅"}