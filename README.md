# Sign Language Production — Phase 1: Text → Pose

> Generating continuous 3D sign language pose sequences from spoken English text, without gloss intermediate representation.

---

## Overview

This repository contains the Phase 1 implementation of an **American Sign Language (ASL) production pipeline** that maps raw English text directly to skeleton pose sequences. We design, implement, and compare three architectures of increasing complexity, all trained and evaluated on the **How2Sign** dataset.

The project is divided into two phases:
- **Phase 1 (this repo):** Text → 3D Pose Sequences
- **Phase 2 (upcoming):** Pose Sequences → Photorealistic Video

---

## Architectures

| | Approach | Strategy | Inspired by |
|---|---|---|---|
| **A** | Transformer Seq2Seq | Autoregressive, frame-by-frame | Fast-SLP (SignDiff) |
| **B** | Transformer + Diffusion | DDPM conditioned on T5 embeddings | MaDiS / SignDiff |
| **C** | Transformer + Flow Matching | OT-CFM, ODE integration | SignFlow |

All three approaches share the same **T5-small** text encoder and are evaluated on identical splits for fair comparison.

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
├── data/
│   ├── dataset.py          # How2SignDataset — PyTorch Dataset class
│   ├── collate.py          # collate_fn with dynamic padding
│   ├── normalize.py        # Normalization stats computation
│   └── stats.pt            # Pre-computed mean/std (from train set)
│
├── models/
│   ├── approach_a/
│   │   └── transformer_seq2seq.py
│   ├── approach_b/
│   │   └── diffusion_model.py
│   └── approach_c/
│       └── flow_matching.py
│
├── eval/
│   ├── metrics.py          # MPJPE, DTW
│   └── visualize.py        # Skeleton sequence visualization
│
├── experiments/
│   ├── configs/            # YAML configs per approach
│   └── results.csv         # Final comparison table
│
├── notebooks/
│   ├── exploration.ipynb
│   ├── train_a.ipynb
│   ├── train_b.ipynb
│   └── train_c.ipynb
│
├── train.py                # Unified training script
├── evaluate.py             # Unified evaluation script
└── requirements.txt
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

Add the How2Sign dataset to your Kaggle notebook from:  
`/kaggle/input/datasets/sarraverse/how2signdataset/`

### 4. Load normalization stats

The `data/stats.pt` file is pre-computed from the train set and included in the repo. No need to recompute it.

### 5. Run training

```bash
# Approach A
python train.py --approach a --config experiments/configs/approach_a.yaml

# Approach B
python train.py --approach b --config experiments/configs/approach_b.yaml

# Approach C
python train.py --approach c --config experiments/configs/approach_c.yaml
```

### 6. Evaluate

```bash
python evaluate.py --approach a --split test
```

---

## Evaluation Metrics

- **MPJPE** (Mean Per Joint Position Error) — average Euclidean distance between predicted and ground truth keypoints, in normalized units
- **DTW** (Dynamic Time Warping) — temporal alignment error between predicted and ground truth sequences

---

## Team

| Member | Role | Approach |
|--------|------|----------|
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
- Text encoder: **T5-small** (60M parameters) — frozen except last 2 layers
- All experiments use the same train/dev/test splits for fair comparison
- Model checkpoints are **not tracked** in this repo — save locally or on Google Drive
