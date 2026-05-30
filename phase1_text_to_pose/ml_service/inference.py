"""
inference.py — Text → Pose inference pipeline
Usage:
    from inference import SignLanguageInference
    infer = SignLanguageInference("weights/model_best.pt", "data/stats.pt")
    poses = infer.generate("hello how are you", n_frames=60)
"""
 
import torch
import numpy as np
from scipy.signal import savgol_filter
from transformers import T5Tokenizer
 
from model import ApproachB, load_model, N_KP, LHAND_KP, RHAND_KP, ROOT_IDX
 
 
class SignLanguageInference:
    """
    End-to-end: raw text string  →  pose array (T, 151, 3)
    """
 
    def __init__(
        self,
        weights_path: str,
        stats_path: str,
        tokenizer_path: str = "t5-small",   # local folder or HF model id
        t5_path: str = None,                 # local T5EncoderModel folder (optional)
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
 
        # 1. Load model
        self.model = load_model(weights_path, t5_path=t5_path, device=self.device)
 
        # 2. Load normalisation stats
        self.stats = torch.load(stats_path, map_location="cpu")
        self.mean  = self.stats["mean"].numpy()   # shape (N_KP, 3)
        self.std   = np.where(
            self.stats["std"].numpy() < 1e-6, 1.0, self.stats["std"].numpy()
        )
 
        # 3. Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        print("✅ SignLanguageInference ready")
 
    # ── public API ────────────────────────────────────────────────────────────
 
    def generate(
        self,
        text: str,
        n_frames: int = 60,
        guidance_scale: float = 3.0,
        smooth: bool = True,
    ) -> np.ndarray:
        """
        Args:
            text:           input gloss/sentence
            n_frames:       how many frames to generate
            guidance_scale: CFG strength (3–5 recommended)
            smooth:         apply Savitzky-Golay smoothing to hands
 
        Returns:
            poses: np.ndarray  shape (n_frames, 151, 3)
                   denormalised, confidence channel set to 1.0
                   keypoint layout: body[0:25] | face[25:92] | lhand[92:113] | rhand[113:134]
        """
        # Tokenise
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=200,
        )
        ids  = enc["input_ids"].to(self.device)
        amsk = enc["attention_mask"].to(self.device)
 
        # Run diffusion
        with torch.no_grad():
            out = self.model.generate(ids, amsk, n_frames=n_frames, guidance_scale=guidance_scale)
        poses_norm = out[0].cpu().numpy()           # (n_frames, 151, 3)
 
        # Denormalise
        poses = poses_norm * self.std + self.mean   # (n_frames, 151, 3)
 
        # Confidence channel → 1.0  (z values from model are meaningless)
        poses[:, :, 2] = 1.0
 
        if smooth:
            poses = self._smooth(poses)
 
        return poses
 
    def generate_openpose_json(
        self,
        text: str,
        n_frames: int = 60,
        guidance_scale: float = 3.0,
    ) -> dict:
        """Same as generate() but returns OpenPose-compatible JSON dict."""
        poses = self.generate(text, n_frames, guidance_scale)
        frames = []
        for fi, fr in enumerate(poses):
            frames.append({
                "frame_id": fi,
                "people": [{
                    "person_id":                [-1],
                    "pose_keypoints_2d":        fr[:25].flatten().tolist(),
                    "face_keypoints_2d":        fr[25:92].flatten().tolist(),
                    "hand_left_keypoints_2d":   fr[92:113].flatten().tolist(),
                    "hand_right_keypoints_2d":  fr[113:134].flatten().tolist(),
                }],
            })
        return {
            "text": text,
            "n_frames": len(poses),
            "guidance_scale": guidance_scale,
            "keypoint_layout": "OpenPose 151kp (body25+face67+lhand21+rhand21)",
            "frames": frames,
        }
 
    # ── internal helpers ──────────────────────────────────────────────────────
 
    @staticmethod
    def _smooth(poses: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
        if poses.shape[0] < window:
            return poses
        out = poses.copy()
        for kp in LHAND_KP + RHAND_KP:
            for c in range(2):
                out[:, kp, c] = savgol_filter(out[:, kp, c], window, polyorder)
        return out
 
 
if __name__ == "__main__":
    infer = SignLanguageInference(
        weights_path="weights/model_best.pt",
        stats_path="data/stats.pt",
        tokenizer_path="t5-small",
    )
 
    poses = infer.generate("hello how are you", n_frames=60)
 
    print("Shape:", poses.shape)
    print("First frame:", poses[0, 0])