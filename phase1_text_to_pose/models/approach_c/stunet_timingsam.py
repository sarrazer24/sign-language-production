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
#  PARTIE HIBA — STUNet + VelocityGuidedTimingSAM + MHMC + SAA + CNNBlock
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
#  5. CNN BLOCK
#  Fusionne les features de mouvement avec le conditioning (texte + timestep)
#  puis fait un down/up-sample selon le niveau de l'encoder/decoder
# ─────────────────────────────────────────────

class CNNBlock(nn.Module):
    """
    CNN Block du STUNet (Fig. 2 du papier).
    - Intègre le conditioning z_c (texte + timestep) via scale+shift (AdaGN)
    - Applique GroupNorm + Conv1d + SiLU
    - down=True  → average pooling  (F↓D↓ en encoder)
    - up=True    → nearest interp   (F↑D↑ en decoder)
    """
    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        cond_dim:     int,
        dropout:      float = 0.1,
        down:         bool  = False,
        up:           bool  = False,
        kernel_size:  int   = 3,
    ):
        super().__init__()
        self.down = down
        self.up   = up

        padding = kernel_size // 2

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

        self.norm2   = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)

        # Projection du conditioning → scale + shift
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2),
        )

        # Shortcut si les dims changent
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

        # Resampling
        if down:
            self.resample = nn.AvgPool1d(kernel_size=2, stride=2)
        elif up:
            self.resample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, C, T)   — motion features (C=channels, T=frames)
        z_c : (B, cond_dim) — conditioning (texte + timestep fusionnés)
        → (B, out_channels, T')
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.resample(h)
        h = self.conv1(h)

        # AdaGN : scale + shift depuis le conditioning
        scale, shift = self.cond_proj(z_c).chunk(2, dim=-1)   # (B, out_channels) chacun
        h = self.norm2(h)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        skip = self.resample(self.shortcut(x)) if (self.down or self.up) else self.shortcut(x)
        return h + skip


# ─────────────────────────────────────────────
#  6. MHMC — Multi-Head Mixed Convolution (1D)
#  Kernel sizes : 3, 5, 7, … (2i+1 par tête)
# ─────────────────────────────────────────────

class MHMC(nn.Module):
    """
    Multi-Head Mixed Convolution adapté pour les séquences 1D.
    Divise les channels en N têtes, chaque tête a un kernel différent.
    Kernel k_i = 2i + 1  → têtes avec k = 3, 5, 7 pour N=3
    (Algorithm 2 du papier, procédure MHMC)
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels doit être divisible par num_heads"
        self.num_heads   = num_heads
        self.head_dim    = channels // num_heads

        # Une conv dépthwise séparable par tête, kernel 3/5/7
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(
                self.head_dim, self.head_dim,
                kernel_size = 2 * i + 3,    # 3, 5, 7, …
                padding     = i + 1,         # same padding
                groups      = self.head_dim, # depthwise
            )
            for i in range(num_heads)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : (B, C, T)
        → (B, C, T)  — même shape, features multi-échelles
        """
        heads = torch.chunk(h, self.num_heads, dim=1)  # N × (B, head_dim, T)
        out   = [conv(hi) for conv, hi in zip(self.dw_convs, heads)]
        return torch.cat(out, dim=1)                   # (B, C, T)


# ─────────────────────────────────────────────
#  7. SAA — Scale-Aware Aggregation
#  Shuffle + groupement cross-échelles + inverse-bottleneck
# ─────────────────────────────────────────────

class SAA(nn.Module):
    """
    Scale-Aware Aggregation (Algorithm 2, procédure SAA).
    - Divise en M groupes (un canal par tête → M groupes de N canaux)
    - 1×1 LightWeight conv + 1×1 Conv par groupe
    - Retourne la carte d'attention Q de même shape que l'entrée
    """
    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = None):
        super().__init__()
        self.num_heads  = num_heads
        # M groupes = head_dim (chaque groupe regroupe 1 canal de chaque tête)
        self.num_groups = num_groups or (channels // num_heads)
        group_size      = num_heads    # canaux par groupe = N têtes

        # LightWeight 1×1 conv + 1×1 conv pour chaque groupe
        self.lw_convs  = nn.ModuleList([
            nn.Conv1d(group_size, group_size, kernel_size=1, groups=group_size)
            for _ in range(self.num_groups)
        ])
        self.pw_convs  = nn.ModuleList([
            nn.Conv1d(group_size, group_size, kernel_size=1)
            for _ in range(self.num_groups)
        ])

    def forward(self, H_g: torch.Tensor) -> torch.Tensor:
        """
        H_g : (B, C, T)  — sortie du Grouping (fusionnée fwd+bwd)
        → Q : (B, C, T)  — carte d'attention
        """
        B, C, T = H_g.shape
        # Reshape → (B, num_groups, num_heads, T) pour grouper 1 canal/tête
        x = H_g.view(B, self.num_groups, self.num_heads, T)

        q_list = []
        for i in range(self.num_groups):
            g_i = x[:, i, :, :]           # (B, num_heads, T)
            q_i = self.lw_convs[i](g_i)   # LW 1×1
            q_i = self.pw_convs[i](q_i)   # 1×1
            q_list.append(q_i)

        Q = torch.stack(q_list, dim=1)    # (B, num_groups, num_heads, T)
        return Q.view(B, C, T)            # (B, C, T)


# ─────────────────────────────────────────────
#  8. VELOCITY-GUIDED TIMING-SAM
#  Contribution de l'équipe : on passe [X, Vel(X)] au BiGRU
#  → input_dim du BiGRU = 2 * channels (au lieu de channels)
# ─────────────────────────────────────────────

class VelocityGuidedTimingSAM(nn.Module):
    """
    Timing-SAM avec Velocity Injection (notre contribution, Approche C).

    Différence vs SignSAM original :
      - Input du BiGRU : concat([X, Vel(X)]) → dim 2*channels
      - Le BiGRU voit positions ET vitesses
      - Tout le reste (MHMC, Grouping, SAA, modulation Z = Q ⊙ V) est identique

    Algorithm 2 du papier (avec cette seule modification).
    """
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.channels  = channels
        self.num_heads = num_heads

        # BiGRU : input = 2*channels (X concat Vel(X)), hidden = channels
        self.bigru = nn.GRU(
            input_size  = 2 * channels,
            hidden_size = channels,
            num_layers  = 1,
            batch_first = True,
            bidirectional = True,
        )

        # MHMC indépendants pour fwd et bwd
        self.mhmc_fwd = MHMC(channels, num_heads)
        self.mhmc_bwd = MHMC(channels, num_heads)

        # SAA sur la fusion fwd+bwd
        self.saa = SAA(channels, num_heads)

        # Projection valeur (Wv)
        self.Wv = nn.Conv1d(channels, channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def compute_velocity(x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, C, T)
        vel : (B, C, T) — vel[:, :, t] = x[:, :, t+1] - x[:, :, t], zéro à t=T-1
        """
        vel = torch.zeros_like(x)
        vel[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
        return vel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)  — fusion features mouvement + conditioning (sortie CNNBlock)
        → Z : (B, C, T)
        """
        B, C, T = x.shape

        # ── Velocity injection ──────────────────────────────────────────
        vel = self.compute_velocity(x)                          # (B, C, T)
        x_vel = torch.cat([x, vel], dim=1)               # (B, 2C, T)

        # ── BiGRU  (batch_first → swap T et C) ──────────────────────────
        x_vel_t = x_vel.permute(0, 2, 1)                 # (B, T, 2C)
        h, _    = self.bigru(x_vel_t)                    # (B, T, 2C)  [fwd||bwd]
        h       = self.dropout(h)

        H_fwd = h[:, :, :C].permute(0, 2, 1)            # (B, C, T) — forward
        H_bwd = h[:, :, C:].permute(0, 2, 1)            # (B, C, T) — backward

        # ── MHMC indépendants ───────────────────────────────────────────
        H_fwd_ms = self.mhmc_fwd(H_fwd)                  # (B, C, T)
        H_bwd_ms = self.mhmc_bwd(H_bwd)                  # (B, C, T)

        # ── Grouping + fusion ────────────────────────────────────────────
        # Regroupe 1 canal de chaque tête (comme Algorithm 2 / procédure Grouping)
        head_dim = C // self.num_heads
        fwd_groups = torch.stack(torch.chunk(H_fwd_ms, self.num_heads, dim=1), dim=2)  # (B, head_dim, N, T)
        bwd_groups = torch.stack(torch.chunk(H_bwd_ms, self.num_heads, dim=1), dim=2)

        # Somme élément par élément fwd + bwd (eq. 9 du papier)
        H_g = (fwd_groups + bwd_groups).view(B, C, T)    # (B, C, T)

        # ── SAA → carte d'attention Q ────────────────────────────────────
        Q = self.saa(H_g)                                 # (B, C, T)

        # ── Modulation finale : Z = Q ⊙ (Wv · X) ───────────────────────
        V = self.Wv(x)                                    # (B, C, T)
        Z = Q * V                                         # (B, C, T)

        return Z


# ─────────────────────────────────────────────
#  9. STUNET
#  Encoder (downsampling F↓D↓) + Decoder (upsampling F↑D↑)
#  Skip connections encoder → decoder au même niveau
#  Chaque niveau = CNNBlock + VelocityGuidedTimingSAM
# ─────────────────────────────────────────────

class STUNet(nn.Module):
    """
    Space-Time UNet adapté pour la SLP (Fig. 2 & 3c du papier + Algorithm 1).

    Architecture (3 niveaux) :
      Encoder :
        level 0 : CNNBlock(in→C,   down=False) + TimingSAM   → skip_0
        level 1 : CNNBlock(C→2C,   down=True)  + TimingSAM   → skip_1
        level 2 : CNNBlock(2C→4C,  down=True)  + TimingSAM   → skip_2  (bottleneck)
      Decoder :
        level 2→1 : skip_2+skip_2=skip  → CNNBlock(4C→2C, up=True)  + TimingSAM
        level 1→0 : skip + skip_1       → CNNBlock(4C→C,  up=True)  + TimingSAM
        level 0   : skip + skip_0       → CNNBlock(2C→C,  up=False)

      Projection finale : C → K*3   (K=n_keypoints)

    Le conditioning z_c = t_embed + c_text est injecté dans chaque CNNBlock.
    """
    def __init__(
        self,
        n_keypoints:    int   = N_KEYPOINTS,
        model_channels: int   = 512,
        cond_dim:       int   = 512,
        dropout:        float = 0.1,
        num_heads:      int   = 4,
        kernel_size:    int   = 3,
    ):
        super().__init__()
        C   = model_channels
        dim = cond_dim

        # ── Conditioning : timestep embedding → même dim que cond ──────
        self.time_embed = TimestepMLP(dim)

        # Projection initiale  K*3 → C  (aplatit keypoints)
        self.input_proj = nn.Conv1d(n_keypoints * 3, C, kernel_size=1)

        # ── Encoder ─────────────────────────────────────────────────────
        # level 0 : pas de downsampling (resolution pleine)
        self.enc0_cnn = CNNBlock(C,    C,    cond_dim=dim, dropout=dropout,
                                 down=False, kernel_size=kernel_size)
        self.enc0_sam = VelocityGuidedTimingSAM(C,    num_heads=num_heads)

        # level 1 : downsampling ×2
        self.enc1_cnn = CNNBlock(C,    2*C,  cond_dim=dim, dropout=dropout,
                                 down=True,  kernel_size=kernel_size)
        self.enc1_sam = VelocityGuidedTimingSAM(2*C,  num_heads=num_heads)

        # level 2 (bottleneck) : downsampling ×2
        self.enc2_cnn = CNNBlock(2*C,  4*C,  cond_dim=dim, dropout=dropout,
                                 down=True,  kernel_size=kernel_size)
        self.enc2_sam = VelocityGuidedTimingSAM(4*C,  num_heads=num_heads)

        # ── Decoder ─────────────────────────────────────────────────────
        # level 2→1 : skip = enc2 + enc2  (4C+4C=8C en entrée car skip conn.)
        self.dec2_cnn = CNNBlock(4*C + 4*C, 2*C, cond_dim=dim, dropout=dropout,
                                 up=True, kernel_size=kernel_size)
        self.dec2_sam = VelocityGuidedTimingSAM(2*C, num_heads=num_heads)

        # level 1→0 : skip = dec2 + enc1  (2C+2C=4C)
        self.dec1_cnn = CNNBlock(2*C + 2*C, C,   cond_dim=dim, dropout=dropout,
                                 up=True, kernel_size=kernel_size)
        self.dec1_sam = VelocityGuidedTimingSAM(C,   num_heads=num_heads)

        # level 0 : skip = dec1 + enc0  (C+C=2C)
        self.dec0_cnn = CNNBlock(C + C,     C,   cond_dim=dim, dropout=dropout,
                                 up=False, kernel_size=kernel_size)
        # pas de TimingSAM au dernier niveau (optionnel, allège le modèle)

        # ── Projection finale ────────────────────────────────────────────
        self.output_norm = nn.GroupNorm(8, C)
        self.output_proj = nn.Conv1d(C, n_keypoints * 3, kernel_size=1)

    def forward(
        self,
        x_noisy: torch.Tensor,   # (B, T, K, 3)
        t:       torch.Tensor,   # (B,)  indices temporels diffusion
        c_text:  torch.Tensor,   # (B, cond_dim)  embedding texte (T5)
    ) -> torch.Tensor:
        """
        Prédit x0 à partir du bruit x_t, guidé par c_text et le pas t.
        Retourne (B, T, K, 3).
        """
        B, T, K, _ = x_noisy.shape

        # ── Aplatir les keypoints ────────────────────────────────────────
        x = x_noisy.reshape(B, T, K * 3)          # (B, T, K*3)
        x = x.permute(0, 2, 1)                    # (B, K*3, T)  pour Conv1d

        # ── Conditioning z_c = t_embed + c_text ─────────────────────────
        t_emb = get_timestep_embedding(t, c_text.shape[-1])  # (B, dim)
        t_emb = self.time_embed(t_emb)                       # (B, dim)
        z_c   = t_emb + c_text                               # (B, dim)

        # ── Input projection ─────────────────────────────────────────────
        h = self.input_proj(x)                    # (B, C, T)

        # ── Encoder ─────────────────────────────────────────────────────
        # Level 0
        h = h + self.enc0_cnn(h, z_c)  # (B, C, T)
        s0 = self.enc0_sam(h)           # skip_0 (B, C, T)

        # Level 1  
        h = self.enc1_cnn(s0, z_c)     # (B, 2C, T/2)  ← pas de += ici (dim change!)
        s1 = self.enc1_sam(h)           # skip_1 (B, 2C, T/2)

        # Level 2
        h = self.enc2_cnn(s1, z_c)     # (B, 4C, T/4)  ← idem
        s2 = self.enc2_sam(h)           # skip_2 (B, 4C, T/4)
        # ── Decoder ─────────────────────────────────────────────────────
        # Resize s2 pour le skip si nécessaire (normalement même shape)
        h = self.dec2_cnn(torch.cat([s2, s2], dim=1), z_c)
        h = self.dec2_sam(h)                      # (B, 2C, T/2)

        # Align T si le downsampling a créé un décalage d'1 frame
        if h.shape[-1] != s1.shape[-1]:
            h = F.interpolate(h, size=s1.shape[-1], mode='nearest')
        h = self.dec1_cnn(torch.cat([h, s1], dim=1), z_c)
        h = self.dec1_sam(h)                      # (B, C, T)

        if h.shape[-1] != s0.shape[-1]:
            h = F.interpolate(h, size=s0.shape[-1], mode='nearest')
        h = self.dec0_cnn(torch.cat([h, s0], dim=1), z_c)             # (B, C, T)

        # ── Output projection ────────────────────────────────────────────
        h = F.silu(self.output_norm(h))
        h = self.output_proj(h)                   # (B, K*3, T)

        # ── Remettre en forme ────────────────────────────────────────────
        h = h.permute(0, 2, 1)                    # (B, T, K*3)
        return h.reshape(B, T, K, 3)              # (B, T, K, 3)
    
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
    
    # ── Test partie Hiba ──────────────────────────────────────────────
    B, T, K = 2, 64, N_KEYPOINTS
    C = 256

    x = torch.randn(B, T, K, 3).to(device)
    t = torch.randint(0, 50, (B,)).to(device)
    c = torch.randn(B, C).to(device)

    net = STUNet(n_keypoints=K, model_channels=C, cond_dim=C).to(device)
    out = net(x, t, c)
    print(f"STUNet output    : {out.shape}")   # (2, 64, 151, 3)
    assert out.shape == (B, T, K, 3), "Shape incorrecte !"
    print("✓ Partie Hiba OK")

    # ── Test modèle complet ───────────────────────────────────────────
    model = SignSAM_C(n_keypoints=N_KEYPOINTS, model_channels=512).to(device)
    ids2  = torch.randint(0, 100, (2, 20)).to(device)
    mask2 = torch.ones(2, 20, dtype=torch.long).to(device)
    x_in  = torch.randn(2, 64, N_KEYPOINTS, 3).to(device)
    t_in  = torch.randint(0, 50, (2,)).to(device)
    out2  = model(x_in, t_in, ids2, mask2)
    print(f"SignSAM_C output : {out2.shape}")  # (2, 64, 151, 3)
    assert out2.shape == (2, 64, N_KEYPOINTS, 3)
    print("✓ Modèle complet OK")