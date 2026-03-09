# Sign Language Production — Text to Video

> End-to-end pipeline for generating photorealistic American Sign Language (ASL) videos from spoken English text, without gloss intermediate representation.

---

## Overview

This repository implements a full **two-phase ASL production pipeline**:

| Phase | Input → Output | Status |
|-------|---------------|--------|
| **Phase 1** | English Text → 3D Pose Sequences | 🔄 In progress |
| **Phase 2** | 3D Pose Sequences → Photorealistic Video | 🔜 Upcoming |

The two phases are designed to be modular — Phase 2 takes as input the pose sequences produced by Phase 1, allowing each phase to be developed, evaluated, and improved independently.

---

## Phase 1 — Text → Pose

We design, implement, and compare **three architectures** of increasing complexity, all trained on the **How2Sign** dataset with a shared **T5-small** text encoder.

| | Approach | Strategy | Inspired by |
|---|---|---|---|
| **A** | Transformer Seq2Seq | Autoregressive, frame-by-frame | Fast-SLP (SignDiff) |
| **B** | Transformer + Diffusion | DDPM conditioned on T5 embeddings | MaDiS / SignDiff |
| **C** | Transformer + Flow Matching | OT-CFM, ODE integration | SignFlow |

The best-performing architecture from Phase 1 will feed into Phase 2.

---

## Phase 2 — Pose → Video *(upcoming)*

Rendering skeleton pose sequences into photorealistic signer videos. Architecture to be defined based on Phase 1 results — likely to draw inspiration from ControlNet-based diffusion approaches (e.g. SignDiff's FR-NET).

---

## Dataset

**How2Sign** — a large-scale ASL dataset of instructional videos.

| Split | Samples |
|-------|---------|
| Train | 31,046 |
| Dev | 1,739 |
| Test | 2,343 |
| **Total** | **35,128** |

Each sample contains three aligned modalities:
- `.skels` — 3D skeleton keypoint sequences (x, y, z per keypoint per frame)
- `.text` — English transcription
- `.files` — video segment reference

The dataset is hosted on Kaggle and is **not included in this repository**.  
👉 [How2Sign on Kaggle](https://www.kaggle.com/datasets/sarraverse/how2signdataset)

---

## Repository Structure

```
sign-language-production/
│
├── phase1_text_to_pose/
│   ├── data/
│   │   ├── dataset.py              # How2SignDataset — PyTorch Dataset class
│   │   ├── collate.py              # collate_fn with dynamic padding
│   │   ├── normalize.py            # Normalization stats computation
│   │   └── stats.pt                # Pre-computed mean/std (from train set)
│   │
│   ├── models/
│   │   ├── approach_a/
│   │   │   └── transformer_seq2seq.py
│   │   ├── approach_b/
│   │   │   └── diffusion_model.py
│   │   └── approach_c/
│   │       └── flow_matching.py
│   │
│   ├── eval/
│   │   ├── metrics.py              # MPJPE, DTW
│   │   └── visualize.py            # Skeleton sequence visualization
│   │
│   ├── experiments/
│   │   ├── configs/                # YAML configs per approach
│   │   │   ├── approach_a.yaml
│   │   │   ├── approach_b.yaml
│   │   │   └── approach_c.yaml
│   │   └── results.csv             # Final comparison table
│   │
│   ├── notebooks/
│   │   ├── exploration.ipynb
│   │   ├── train_a.ipynb
│   │   ├── train_b.ipynb
│   │   └── train_c.ipynb
│   │
│   ├── train.py                    # Unified training script
│   └── evaluate.py                 # Unified evaluation script
│
├── phase2_pose_to_video/           # To be populated in Phase 2
│   └── .gitkeep
│
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/sign-language-production.git
cd sign-language-production
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up the dataset on Kaggle

Add the How2Sign dataset to your Kaggle notebook:  
`/kaggle/input/datasets/sarraverse/how2signdataset/`

### 4. Load normalization stats

`phase1_text_to_pose/data/stats.pt` is pre-computed from the train set and included in the repo — no need to recompute.

### 5. Run Phase 1 training

```bash
cd phase1_text_to_pose

# Approach A — Transformer Seq2Seq (baseline)
python train.py --approach a --config experiments/configs/approach_a.yaml

# Approach B — Diffusion
python train.py --approach b --config experiments/configs/approach_b.yaml

# Approach C — Flow Matching
python train.py --approach c --config experiments/configs/approach_c.yaml
```

### 6. Evaluate

```bash
python evaluate.py --approach a --split test
```

---

## Evaluation Metrics

**Phase 1 — Pose Generation**
- **MPJPE** (Mean Per Joint Position Error) — average Euclidean distance between predicted and ground truth keypoints
- **DTW** (Dynamic Time Warping) — temporal alignment error between predicted and ground truth sequences

**Phase 2 — Video Generation** *(to be defined)*
- FVD (Fréchet Video Distance)
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)

---

## Team

| Member | Role | Phase 1 |
|--------|------|---------|
| **Sarra** | Project lead + Data pipeline | Approach B (Diffusion) |
| **Sara** | Implementation | Approach C (Flow Matching) |
| **Hiba** | Implementation + Evaluation | Approach C (Flow Matching) |
| **Nafissa** | Implementation | Approach A (Seq2Seq) |
| **Serine** | Implementation | Approach A (Seq2Seq) |

---

## References

- **SignDiff** — Sign Language Production via Diffusion
- **MaDiS** — Masked Diffusion for Sign Language Production
- **SignFlow** — Flow Matching for Sign Language Production
- **How2Sign** — A Large-scale Multimodal Dataset for Continuous American Sign Language

---

## Notes

- Skeleton sequences are truncated to **500 frames** for initial experiments
- Text encoder: **T5-small** (60M params), frozen except last 2 layers
- All Phase 1 experiments use identical train/dev/test splits for fair comparison
- Model checkpoints are **not tracked** in this repo — save locally or on Google Drive
