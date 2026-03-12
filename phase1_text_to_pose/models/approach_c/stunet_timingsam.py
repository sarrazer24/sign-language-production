"""
stunet_timingsam.py — Approche C
==================================
Répartition :
  Sara  → T5TextEncoder, GaussianDiffusion (DDIM), reconstruction_loss
  Hiba  → STUNet, VelocityGuidedTimingSAM, MHMC, SAA, CNNBlock

N_KEYPOINTS = 151  (How2Sign, confirmé dans les instructions)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel

N_KEYPOINTS = 151   # How2Sign — ne pas modifier


# ═══════════════════════════════════════════════════════════════════════════════
#  PARTIE SARA — T5 + Diffusion DDIM + Loss
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
#  1. TEXT ENCODER (T5-small)
# ─────────────────────────────────────────────

class T5TextEncoder(nn.Module):
    """
    Encodeur T5-small.
    - Tout gelé sauf les 2 dernières couches (instructions)
    - Reçoit input_ids + attention_mask déjà tokenisés par le Dataset
    - Retourne un vecteur fixe par phrase via mean-pooling : (B, out_dim)
    """
    def __init__(self, out_dim: int = 512, model_name: str = "t5-small"):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)

        # Geler tout sauf les 2 derniers blocs
        for layer in list(self.encoder.encoder.block)[:-2]:
            for p in layer.parameters():
                p.requires_grad = False

        # Projection vers out_dim
        d_model   = self.encoder.config.d_model   # 512 pour t5-small
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids      : (B, L)
        attention_mask : (B, L)
        → (B, out_dim)
        """
        out = self.encoder(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        ).last_hidden_state                           # (B, L, d_model)

        # Mean-pool sur les tokens non-paddés
        mask_f = attention_mask.unsqueeze(-1).float() # (B, L, 1)
        pooled = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)

        return self.proj(pooled)                      # (B, out_dim)


# ─────────────────────────────────────────────
#  2. TIMESTEP EMBEDDING
# ─────────────────────────────────────────────

def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoïdal embedding. timesteps : (B,) → (B, dim)"""
    assert dim % 2 == 0
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) *
        torch.arange(half, dtype=torch.float32, device=timesteps.device) / (half - 1)
    )
    args = timesteps.float()[:, None] * freqs[None]   # (B, half)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # (B, dim)


class TimestepMLP(nn.Module):
    """Projette l'embedding sinusoïdal vers la dimension du modèle."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
    def forward(self, x): return self.net(x)


# ─────────────────────────────────────────────
#  3. DIFFUSION DDIM
# ─────────────────────────────────────────────

def _cosine_beta_schedule(T: int, s: float = 0.008):
    import numpy as np
    steps      = T + 1
    x          = np.linspace(0, T, steps)
    ab         = np.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    ab         = ab / ab[0]
    betas      = 1 - ab[1:] / ab[:-1]
    return np.clip(betas, 0, 0.999).astype("float32"), ab[1:].astype("float32")


class GaussianDiffusion:
    """
    Processus de diffusion DDIM.
      - T = 50 steps (instructions Approche C)
      - Scheduler cosine
      - Classifier-free guidance scale = 5.5

    Méthodes :
      q_sample      → forward diffusion (ajout de bruit)  — utilisé à l'entraînement
      ddim_sample   → génération complète                  — utilisé à l'inférence
    """

    def __init__(self, T: int = 50, device: str = "cpu"):
        import numpy as np
        self.T      = T
        self.device = device

        betas, ab      = _cosine_beta_schedule(T)
        ab_prev        = np.append(1.0, ab[:-1])

        def _t(x): return torch.tensor(x, dtype=torch.float32, device=device)

        self.sqrt_ab      = _t(np.sqrt(ab))
        self.sqrt_1_ab    = _t(np.sqrt(1.0 - ab))
        self.alphas_bar   = _t(ab)
        self.alphas_bar_prev = _t(ab_prev)

    # ── Forward process ──────────────────────────────────────────────────────

    def q_sample(
        self,
        x0:    torch.Tensor,          # (B, T, K, 3)  poses propres
        t:     torch.Tensor,          # (B,)  indices entiers
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """Ajoute du bruit gaussien à x0 selon le pas t. Retourne x_t."""
        if noise is None:
            noise = torch.randn_like(x0)
        s_ab  = self.sqrt_ab[t][:, None, None, None]
        s_1ab = self.sqrt_1_ab[t][:, None, None, None]
        return s_ab * x0 + s_1ab * noise

    # ── Inférence DDIM ───────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model,                          # SignSAM_C
        input_ids:      torch.Tensor,   # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
        shape:          tuple,          # (B, T, K, 3)
        guidance_scale: float = 5.5,
        device:         str   = "cpu",
    ) -> torch.Tensor:
        """
        Génère des séquences de poses à partir de bruit pur.
        Utilise classifier-free guidance :
            x_pred = x_uncond + scale * (x_cond - x_uncond)
        Retourne x0 : (B, T, K, 3)
        """
        B = shape[0]
        x = torch.randn(shape, device=device)

        # Pas de temps en ordre décroissant : T-1, T-2, ..., 0
        t_seq = list(range(self.T - 1, -1, -1))

        # Tokens vides pour la branche non-conditionnée
        empty_ids  = torch.zeros_like(input_ids)
        empty_mask = torch.zeros_like(attention_mask)

        for t_val in t_seq:
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Prédictions conditionnelle et non-conditionnelle
            x_cond   = model(x, t_batch, input_ids,  attention_mask)
            x_uncond = model(x, t_batch, empty_ids,  empty_mask)

            # Classifier-free guidance
            x_pred = x_uncond + guidance_scale * (x_cond - x_uncond)

            # Mise à jour DDIM
            ab      = self.alphas_bar[t_val]
            ab_prev = self.alphas_bar_prev[t_val]
            eps     = (x - torch.sqrt(ab) * x_pred) / (torch.sqrt(1.0 - ab) + 1e-8)
            x       = torch.sqrt(ab_prev) * x_pred + torch.sqrt(1.0 - ab_prev) * eps

        return x   # (B, T, K, 3)


# ─────────────────────────────────────────────
#  4. LOSS  (Smooth-L1 poses + Smooth-L1 vélocités)
# ─────────────────────────────────────────────

def reconstruction_loss(
    pred:   torch.Tensor,        # (B, T, K, 3)
    target: torch.Tensor,        # (B, T, K, 3)
    mask:   torch.Tensor = None, # (B, T)  1=valide, 0=padding
) -> torch.Tensor:
    """
    Loss = Smooth-L1(poses) + Smooth-L1(vélocités)
    Identique à l'équation 10 de SignSAM.
    Le masque ignore les frames de padding.
    """

    def velocity(x: torch.Tensor) -> torch.Tensor:
        """Vel(X)^i = x^{i+1} - x^i, zéro sur la dernière frame."""
        v = torch.zeros_like(x)
        v[:, :-1] = x[:, 1:] - x[:, :-1]
        return v

    def masked_smooth_l1(a, b, mask):
        # (B, T, K, 3) → loss scalaire
        loss = F.smooth_l1_loss(a, b, reduction='none')  # (B, T, K, 3)
        loss = loss.mean(dim=(-2, -1))                    # (B, T)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp(min=1e-8)
        else:
            loss = loss.mean()
        return loss

    pos_loss = masked_smooth_l1(pred, target, mask)
    vel_loss = masked_smooth_l1(velocity(pred), velocity(target), mask)

    return pos_loss + vel_loss


# ═══════════════════════════════════════════════════════════════════════════════
#  PARTIE HIBA — STUNet + VelocityGuidedTimingSAM
#  (à compléter par Hiba)
# ═══════════════════════════════════════════════════════════════════════════════

class CNNBlock(nn.Module):
    # TODO Hiba
    pass


class MHMC(nn.Module):
    # TODO Hiba
    pass


class SAA(nn.Module):
    # TODO Hiba
    pass


class VelocityGuidedTimingSAM(nn.Module):
    # TODO Hiba
    pass


class STUNet(nn.Module):
    # TODO Hiba
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  MODÈLE COMPLET — assemblage Sara + Hiba
# ═══════════════════════════════════════════════════════════════════════════════

class SignSAM_C(nn.Module):
    """
    Modèle complet Approche C.
    T5TextEncoder (Sara) + STUNet/Timing-SAM (Hiba).
    """
    def __init__(
        self,
        n_keypoints:    int   = N_KEYPOINTS,
        model_channels: int   = 512,
        dropout:        float = 0.1,
        t5_model:       str   = "t5-small",
    ):
        super().__init__()
        cond_dim          = model_channels
        self.text_encoder = T5TextEncoder(out_dim=cond_dim, model_name=t5_model)
        self.denoiser     = STUNet(
            n_keypoints    = n_keypoints,
            model_channels = model_channels,
            cond_dim       = cond_dim,
            dropout        = dropout,
        )

    def forward(
        self,
        x_noisy:        torch.Tensor,   # (B, T, K, 3)
        t:              torch.Tensor,   # (B,)
        input_ids:      torch.Tensor,   # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        c_text = self.text_encoder(input_ids, attention_mask)   # (B, cond_dim)
        return self.denoiser(x_noisy, t, c_text)                # (B, T, K, 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST RAPIDE  (python stunet_timingsam.py)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test uniquement la partie Sara (avant que Hiba ait fini STUNet)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # ── T5 encoder ──
    encoder = T5TextEncoder(out_dim=512).to(device)
    ids  = torch.randint(0, 100, (2, 20)).to(device)
    mask = torch.ones(2, 20, dtype=torch.long).to(device)
    c    = encoder(ids, mask)
    print(f"T5 output shape  : {c.shape}")   # (2, 512)

    # ── Diffusion ──
    diff  = GaussianDiffusion(T=50, device=device)
    x0    = torch.randn(2, 64, N_KEYPOINTS, 3).to(device)
    t_idx = torch.randint(0, 50, (2,)).to(device)
    x_t   = diff.q_sample(x0, t_idx)
    print(f"q_sample output  : {x_t.shape}")   # (2, 64, 151, 3)

    # ── Loss ──
    loss = reconstruction_loss(x_t, x0)
    print(f"Loss             : {loss.item():.4f}")

    print("\n✓ Partie Sara OK — en attente de la partie Hiba (STUNet)")