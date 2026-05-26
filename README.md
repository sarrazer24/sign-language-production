# Sign Language Production — Text to Video

> End-to-end pipeline for generating photorealistic American Sign Language (ASL) videos from spoken English text, without gloss intermediate representation.

---

## Overview

This repository implements a full **two-phase ASL production pipeline**:

| Phase | Input → Output | Status |
|-------|---------------|--------|
| **Phase 1** | English Text → 3D Pose Sequences | 🔄 In progress |
| **Phase 2** | 3D Pose Sequences → Photorealistic Video | 🔜 Upcoming |

The two phases are modular — Phase 2 takes as input the pose sequences produced by Phase 1, allowing each phase to be developed, evaluated, and improved independently.

---

## Research Contribution

Existing works (SignDiff, MaDiS, SignSAM) encode temporal dynamics **only in the loss function**, never in the model's conditioning. Our central research angle:

> **"We propose temporally-aware conditioning strategies for sign language pose generation, applied across three architectural paradigms."**

Each architecture carries its own original contribution on top of its inspiration:

| | Approach | Inspired by | Our Contribution |
|---|---|---|---|
| **A** | Transformer Seq2Seq | Fast-SLP (SignDiff) | **Duration-Aware Positional Encoding** |
| **B** | Transformer + Diffusion | MaDiS / SignDiff | **Semantic Timestep Conditioning** |
| **C** | STUNet + Timing-SAM | SignSAM | **Velocity-Guided Timing-SAM** |

---

## Phase 1 — Text → Pose

### Approach A — Transformer Seq2Seq *(Nafissa + Serine)*
Standard encoder-decoder Transformer with autoregressive pose generation.

**Original contribution — Duration-Aware Positional Encoding:**  
Instead of a fixed PE, we scale positional embeddings dynamically according to the ratio between text length and predicted pose length:

$$PE_{aware}(pos) = PE(pos) \cdot \sigma\!\left(W \cdot \frac{L_{text}}{L_{pose}}\right)$$

The model becomes aware from the start that a short sentence produces a short sequence, and a long sentence produces a long one.

---

### Approach B — Transformer + Diffusion *(Sarra)*
DDPM with a 1D temporal U-Net conditioned on T5 text embeddings.

**Original contribution — Semantic Timestep Conditioning:**  
Instead of naively summing $t_{embd} + c_{embd}$ (as in SignSAM), we compute an adaptive condition vector via cross-attention between the timestep and the text embeddings:

$$z_c = \text{Attention}(Q = t_{embd},\ K = c_{embd},\ V = c_{embd})$$

At high noise levels (large $t$), the model attends to global sign structure. At low noise levels (small $t$), it attends to fine-grained finger details. The model learns this weighting automatically.

---

### Approach C — STUNet + Timing-SAM *(Sara + Hiba)*
Diffusion with a spatio-temporal U-Net and multi-scale temporal modulation, inspired by SignSAM. Uses DDIM (50 steps) for faster inference.

**Original contribution — Velocity-Guided Timing-SAM:**  
SignSAM computes $Vel(X)$ only for the loss, never to guide the internal attention. We inject velocities directly into the BiGRU of the Timing-SAM module. Instead of passing $X \in \mathbb{R}^{F \times d}$, we pass $[X,\ Vel(X)] \in \mathbb{R}^{F \times 2d}$:

$$Vel(X)^i = x^{i+1} - x^i$$

The BiGRU sees not only *where* keypoints are, but also *how they move* — better capture of transitions between signs.

---

### Architecture Comparison

| | Approach A | Approach B | Approach C |
|---|---|---|---|
| **Generation** | Autoregressive | DDPM (100 steps) | DDIM (50 steps) |
| **Backbone** | Transformer | 1D U-Net | STUNet (joint F+D) |
| **Temporal modeling** | Standard attention | Cross-attention | Timing-SAM + BiGRU |
| **Loss** | Masked MSE | MSE on noise ε | Smooth L1 + velocity |
| **Our contribution** | Duration-Aware PE | Semantic Timestep | Velocity BiGRU |

> **Note on text encoder:** SignSAM originally uses CLIP+BERT. We use **T5-small** for all three approaches to ensure fair comparison — isolating the effect of the generation architecture.

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
│   │   │   └── transformer_seq2seq.py     # Nafissa + Serine
│   │   ├── approach_b/
│   │   │   └── diffusion_model.py         # Sarra
│   │   └── approach_c/
│   │       └── stunet_timingsam.py        # Sara + Hiba
│   │
│   ├── eval/
│   │   ├── metrics.py              # MPJPE, DTW
│   │   └── visualize.py            # Skeleton sequence visualization
│   │
│   ├── experiments/
│   │   ├── configs/
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
│   ├── train.py
│   └── evaluate.py
│
├── phase2_pose_to_video/
│   └── .gitkeep
│
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/sarrazer24/sign-language-production.git
cd sign-language-production
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up the dataset on Kaggle

```
/kaggle/input/datasets/sarraverse/how2signdataset/
```

### 4. Run Phase 1 training

```bash
cd phase1_text_to_pose

python train.py --approach a --config experiments/configs/approach_a.yaml
python train.py --approach b --config experiments/configs/approach_b.yaml
python train.py --approach c --config experiments/configs/approach_c.yaml
```

### 5. Evaluate

```bash
python evaluate.py --approach b --split test
```

---

## Evaluation Metrics

**Phase 1**
- **MPJPE** — Mean Per Joint Position Error
- **DTW** — Dynamic Time Warping distance

**Phase 2** *(upcoming)*
- FVD, SSIM, PSNR

---

## Team

| Member | Role | Phase 1 |
|--------|------|---------|
| **Sarra** | Project lead + Data pipeline | Approach B — Diffusion + Semantic Timestep Conditioning |
| **Sara** | Implementation | Approach C — STUNet + Velocity-Guided Timing-SAM |
| **Hiba** | Implementation | Approach C — STUNet + Velocity-Guided Timing-SAM |
| **Nafissa** | Implementation | Approach A — Seq2Seq + Duration-Aware PE |
| **Serine** | Implementation | Approach A — Seq2Seq + Duration-Aware PE |

---

## References

- **SignDiff** — Sign Language Production via Diffusion Models
- **MaDiS** — Masked Diffusion for Sign Language Production
- **SignSAM** — Sign Language Production with Scale-Aware Modulation
- **How2Sign** — A Large-scale Multimodal Dataset for Continuous American Sign Language

---

## Notes

- Skeleton sequences truncated to **500 frames** for initial experiments
- Text encoder: **T5-small** (60M params), shared across all approaches, frozen except last 2 layers
- Each approach includes a simple **ablation study** (with vs without the original contribution)
- Model checkpoints are **not tracked** in this repo — save locally or on Google Drive
