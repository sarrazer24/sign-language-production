import numpy as np
import torch

# ── Constantes globales ──────────────────────────────────────────────────────
N_KEYPOINTS = 151
BASE = '/kaggle/input/datasets/sarraverse/how2signdataset'

def compute_normalization_stats(n_keypoints, max_samples=5000):
    all_poses = []
    with open(f"{BASE}/train.skels", 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            vals = np.array(line.strip().split(), dtype=np.float32)
            n_frames = len(vals) // (n_keypoints * 3)
            if n_frames == 0:
                continue
            poses = vals[:n_frames * n_keypoints * 3].reshape(n_frames, n_keypoints, 3)
            all_poses.append(poses[:100])  # max 100 frames par sample

    all_poses = np.concatenate(all_poses, axis=0)  # (N_frames_total, K, 3)
    mean = all_poses.mean(axis=0)  # (K, 3)
    std  = all_poses.std(axis=0)   # (K, 3)

    stats = {
        'mean': torch.FloatTensor(mean),
        'std' : torch.FloatTensor(std)
    }
    save_path = '/kaggle/working/sign-language-production/phase1_text_to_pose/data/stats.pt'
    torch.save(stats, save_path)
    print(f"✅ Stats calculées sur {all_poses.shape[0]:,} frames")
    print(f"   mean shape : {mean.shape}")
    print(f"   std  shape : {std.shape}")
    print(f"   mean global : {mean.mean():.4f}")
    print(f"   std  global : {std.mean():.4f}")
    print(f"\n→ stats.pt sauvegardé dans {save_path}")
    return stats

if __name__ == '__main__':
    stats = compute_normalization_stats(N_KEYPOINTS)