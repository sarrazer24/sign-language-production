import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

# ── Constantes globales ──────────────────────────────────────────────────────
N_KEYPOINTS = 151  # How2Sign : 133 corps + 2×21 mains (MediaPipe)
BASE = '/kaggle/input/how2signdataset'

class How2SignDataset(Dataset):
    def __init__(self, split='train', stats=None, max_frames=500):
        assert split in ['train', 'dev', 'test']
        self.max_frames  = max_frames
        self.n_keypoints = N_KEYPOINTS

        # Charger tout en mémoire (filesystem Kaggle est lent sinon)
        with open(f"{BASE}/{split}.skels", 'r') as f:
            self.skels_lines = f.readlines()
        with open(f"{BASE}/{split}.text", 'r', encoding='utf-8') as f:
            self.text_lines = f.readlines()
        with open(f"{BASE}/{split}.files", 'r') as f:
            self.files_lines = f.readlines()

        assert len(self.skels_lines) == len(self.text_lines) == len(self.files_lines)

        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.stats = stats

    def __len__(self):
        return len(self.skels_lines)

    def __getitem__(self, idx):
        # ── Squelettes ──────────────────────────────
        vals = np.array(self.skels_lines[idx].strip().split(), dtype=np.float32)
        n_frames = len(vals) // (self.n_keypoints * 3)
        poses = vals[:n_frames * self.n_keypoints * 3].reshape(n_frames, self.n_keypoints, 3)

        if n_frames > self.max_frames:
            poses = poses[:self.max_frames]
            n_frames = self.max_frames

        if self.stats is not None:
            poses = (poses - self.stats['mean'].numpy()) / \
                    (self.stats['std'].numpy() + 1e-8)

        poses = torch.FloatTensor(poses)  # (T, K, 3)

        # ── Texte ────────────────────────────────────
        text = self.text_lines[idx].strip()
        enc  = self.tokenizer(
            text,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=200
        )

        return {
            'poses'         : poses,
            'pose_length'   : n_frames,
            'input_ids'     : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'text'          : text,
            'file_ref'      : self.files_lines[idx].strip()
        }