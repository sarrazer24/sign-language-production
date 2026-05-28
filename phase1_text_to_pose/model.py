"""
model.py — Full ApproachB v23 architecture
Extracted from train_b.ipynb for production inference.
All training-only code removed. Only what's needed to load weights and run generate().
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
# ─── Constants (must match training exactly) ────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_KP      = 151
POSE_DIM  = 453   # N_KP * 3
T_DIFF    = 100
ROOT_IDX  = 8
 
BODY_KP  = list(range(0,   25))
FACE_KP  = list(range(25,  92))
LHAND_KP = list(range(92,  113))
RHAND_KP = list(range(113, 134))
 
FINGER_INDICES_LHAND, FINGER_INDICES_RHAND = [], []
for d in range(5):
    for j in range(1, 4):
        FINGER_INDICES_LHAND.append(92  + 1 + d*4 + j)
        FINGER_INDICES_RHAND.append(113 + 1 + d*4 + j)
FINGERS = FINGER_INDICES_LHAND + FINGER_INDICES_RHAND
 
REGION_WEIGHT = torch.ones(N_KP)
for kp in BODY_KP:              REGION_WEIGHT[kp] = 1.0
for kp in LHAND_KP + RHAND_KP: REGION_WEIGHT[kp] = 6.0
for kp in FINGERS:              REGION_WEIGHT[kp] = 10.0
for kp in FACE_KP:              REGION_WEIGHT[kp] = 0.4
 
 
# ─── Text Encoder (T5-Small) ─────────────────────────────────────────────────
class TextEncoder(nn.Module):
    def __init__(self, model_name="t5-small", output_dim=512, t5_path=None):
        super().__init__()
        from transformers import T5EncoderModel
        src = t5_path if t5_path else model_name
        self.encoder = T5EncoderModel.from_pretrained(src)
 
        # Same freeze pattern as training: all frozen except last 2 blocks
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.encoder.encoder.block[-2:].parameters():
            p.requires_grad = True
 
        h = self.encoder.config.d_model
        self.proj = nn.Linear(h, output_dim) if h != output_dim else nn.Identity()
 
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state)
 
    def pool(self, input_ids, attention_mask):
        tok = self.forward(input_ids, attention_mask)
        mask = attention_mask.unsqueeze(-1).float().to(tok.device)
        return (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
 
 
# ─── Timestep Embedding ───────────────────────────────────────────────────────
class TimestepEmbedding(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )
 
    def sinusoidal(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args  = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
 
    def forward(self, t):
        return self.mlp(self.sinusoidal(t))
 
 
# ─── UNet1D building blocks ───────────────────────────────────────────────────
class BottleneckCrossAttention(nn.Module):
    def __init__(self, feat_dim, text_dim=512, num_heads=8):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, feat_dim) if text_dim != feat_dim else nn.Identity()
        self.attn      = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True, dropout=0.0)
        self.norm      = nn.LayerNorm(feat_dim)
 
    def forward(self, x, c_tokens):
        q  = x.permute(0, 2, 1)
        kv = self.text_proj(c_tokens)
        out, _ = self.attn(q, kv, kv)
        return self.norm(q + out).permute(0, 2, 1)
 
 
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=512, dropout=0.1):
        super().__init__()
        self.conv1     = nn.Conv1d(in_ch, out_ch, 5, padding=2)
        self.conv2     = nn.Conv1d(out_ch, out_ch, 5, padding=2)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.norm1     = nn.GroupNorm(8, out_ch)
        self.norm2     = nn.GroupNorm(8, out_ch)
        self.act       = nn.SiLU()
        self.drop      = nn.Dropout(dropout)
        self.res_proj  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
 
    def forward(self, x, z_c):
        h = self.act(self.norm1(self.conv1(x)))
        sc, sh = self.cond_proj(z_c).chunk(2, dim=-1)
        h = h * (1 + sc.unsqueeze(-1)) + sh.unsqueeze(-1)
        h = self.drop(self.act(self.norm2(self.conv2(h))))
        return h + self.res_proj(x)
 
 
class UNet1D(nn.Module):
    def __init__(self, pose_dim=POSE_DIM, cond_dim=512, base_ch=128, text_dim=512):
        super().__init__()
        C = base_ch
        self.input_proj  = nn.Conv1d(pose_dim, C, 1)
        self.enc1        = ResBlock1D(C,   C,   cond_dim)
        self.down1       = nn.Conv1d(C,   C,   3, stride=2, padding=1)
        self.enc2        = ResBlock1D(C,   C*2, cond_dim)
        self.down2       = nn.Conv1d(C*2, C*2, 3, stride=2, padding=1)
        self.enc3        = ResBlock1D(C*2, C*4, cond_dim)
        self.down3       = nn.Conv1d(C*4, C*4, 3, stride=2, padding=1)
        self.bottleneck  = ResBlock1D(C*4, C*4, cond_dim)
        self.bottleneck_ca = BottleneckCrossAttention(C*4, text_dim, num_heads=8)
        self.up3  = nn.ConvTranspose1d(C*4, C*4, 4, stride=2, padding=1)
        self.up2  = nn.ConvTranspose1d(C*4, C*2, 4, stride=2, padding=1)
        self.up1  = nn.ConvTranspose1d(C*2, C,   4, stride=2, padding=1)
        self.dec3 = ResBlock1D(C*8, C*4, cond_dim)
        self.dec2 = ResBlock1D(C*4, C*2, cond_dim)
        self.dec1 = ResBlock1D(C*2, C,   cond_dim)
        # NOTE: dec3_ca and dec2_ca removed — not present in checkpoint
        self.out  = nn.Conv1d(C, pose_dim, 1)
 
    @staticmethod
    def _match(x, ref):
        return x[:, :, :ref.shape[2]] if x.shape[2] != ref.shape[2] else x
 
    def forward(self, x, z_c, c_tokens):
        x  = self.input_proj(x.transpose(1, 2))
        s1 = self.enc1(x,  z_c);  x = self.down1(s1)
        s2 = self.enc2(x,  z_c);  x = self.down2(s2)
        s3 = self.enc3(x,  z_c);  x = self.down3(s3)
        x  = self.bottleneck(x, z_c)
        x  = self.bottleneck_ca(x, c_tokens)
        u3 = self._match(self.up3(x),  s3)
        d3 = self.dec3(torch.cat([u3, s3], 1), z_c)
        # dec3_ca removed
        u2 = self._match(self.up2(d3), s2)
        d2 = self.dec2(torch.cat([u2, s2], 1), z_c)
        # dec2_ca removed
        u1 = self._match(self.up1(d2), s1)
        return self.out(self.dec1(torch.cat([u1, s1], 1), z_c)).transpose(1, 2)
 
 
# ─── Pose Projector (training-only — kept so weights load; never called in generate()) ──
class PoseProjector(nn.Module):
    def __init__(self, pose_dim=POSE_DIM, hidden=512, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pose_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
        )
 
    def forward(self, poses, mask):
        m = mask.unsqueeze(-1).float()
        return F.normalize(
            (self.net(poses) * m).sum(1) / m.sum(1).clamp(min=1e-9),
            dim=-1
        )
 
 
# ─── Gaussian Diffusion ────────────────────────────────────────────────────────
class GaussianDiffusion(nn.Module):
    def __init__(self, model, T=100):
        super().__init__()
        self.model = model
        self.T     = T
        betas = self._cosine_schedule(T)
        alphas = 1. - betas
        acp    = torch.cumprod(alphas, 0)
        self.register_buffer("betas",          betas)
        self.register_buffer("alphas",         alphas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_acp",       torch.sqrt(acp))
        # Named "sqrt_one_minus" to match checkpoint
        self.register_buffer("sqrt_one_minus", torch.sqrt(1 - acp))
 
    def _cosine_schedule(self, T, s=0.008):
        st = torch.arange(T + 1, dtype=torch.float32)
        f  = torch.cos((st / T + s) / (1 + s) * math.pi / 2) ** 2
        return torch.clamp(1 - f[1:] / f[:-1], 1e-4, 0.9999)
 
    def _predict_x0(self, x_t, z_c, c_tokens):
        return torch.tanh(self.model(x_t, z_c, c_tokens)) * 3.0
 
    @torch.no_grad()
    def p_sample_cfg(self, x_t, t_val, z_cond, z_uncond, c_tokens, null_tokens, guidance_scale):
        B = x_t.shape[0]
        t = torch.full((B,), t_val, device=x_t.device, dtype=torch.long)
        x0_cond   = self._predict_x0(x_t, z_cond,  c_tokens)
        x0_uncond = self._predict_x0(x_t, z_uncond, null_tokens)
        x0_pred   = torch.clamp(x0_uncond + guidance_scale * (x0_cond - x0_uncond), -3., 3.)
        beta  = self.betas[t][:, None, None]
        alpha = self.alphas[t][:, None, None]
        acp   = self.alphas_cumprod[t][:, None, None]
        coef2 = torch.sqrt(1 - acp)
        eps   = (x_t - torch.sqrt(acp) * x0_pred) / (coef2 + 1e-8)
        mean  = torch.clamp((1 / torch.sqrt(alpha)) * (x_t - beta / (coef2 + 1e-8) * eps), -3., 3.)
        if t_val > 0:
            mean = mean + 0.5 * torch.sqrt(beta) * torch.randn_like(x_t)
        return mean
 
 
# ─── ApproachB — the full model ───────────────────────────────────────────────
class ApproachB(nn.Module):
    P_UNCOND = 0.30
 
    def __init__(self, T=T_DIFF, t5_path=None):
        super().__init__()
        self.text_encoder   = TextEncoder(t5_path=t5_path)
        self.timestep_emb   = TimestepEmbedding(dim=512)
        self.pose_projector = PoseProjector(pose_dim=POSE_DIM, hidden=512, out_dim=512)
        # 3-layer cond_proj matching checkpoint (input is 512, not 1024)
        self.cond_proj = nn.Sequential(
            nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 512)
        )
        self.unet      = UNet1D(pose_dim=POSE_DIM, base_ch=128, text_dim=512)
        self.diffusion = GaussianDiffusion(self.unet, T=T)
 
    def get_condition(self, t_embd, c_pooled):
        # Add instead of cat — matches 512-in cond_proj
        return self.cond_proj(t_embd + c_pooled)
 
    @torch.no_grad()
    def generate(self, input_ids, attention_mask, n_frames=55, guidance_scale=3.0):
        self.eval()
        device = next(self.parameters()).device
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        B = input_ids.shape[0]
 
        c_tokens    = self.text_encoder(input_ids, attention_mask)
        c_pooled    = self.text_encoder.pool(input_ids, attention_mask)
        null_tokens = torch.zeros_like(c_tokens)
        null_pooled = torch.zeros_like(c_pooled)
 
        x = torch.randn(B, n_frames, POSE_DIM, device=device) * 0.9
        for t_val in reversed(range(self.diffusion.T)):
            t_t = torch.full((B,), t_val, device=device, dtype=torch.long)
            te  = self.timestep_emb(t_t)
            x   = self.diffusion.p_sample_cfg(
                x, t_val,
                self.get_condition(te, c_pooled),
                self.get_condition(te, null_pooled),
                c_tokens, null_tokens, guidance_scale
            )
        return torch.clamp(x, -3., 3.).reshape(B, n_frames, N_KP, 3)
 
 
def load_model(weights_path: str, t5_path: str = None, device: str = None) -> ApproachB:
    """
    Load ApproachB with saved weights.
 
    Args:
        weights_path: path to model_best.pt
        t5_path:      local path to T5-small model files (optional, downloads if None)
        device:       'cuda' or 'cpu' (auto-detected if None)
 
    Returns:
        model ready for inference
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
 
    model = ApproachB(T=T_DIFF, t5_path=t5_path)
    state = torch.load(weights_path, map_location=device)
 
    # Unwrap training checkpoint if needed
    if "model_state_dict" in state:
        state = state["model_state_dict"]
 
    # strict=False: pose_projector may not be in checkpoint (training-only module)
    missing, unexpected = model.load_state_dict(state, strict=False)
 
    # Only warn about truly unexpected keys — missing pose_projector is fine
    unexpected_real = [k for k in unexpected if "pose_projector" not in k]
    missing_real    = [k for k in missing    if "pose_projector" not in k]
    if missing_real:
        print(f"⚠️  Missing keys (random-init): {missing_real}")
    if unexpected_real:
        print(f"⚠️  Unexpected keys (ignored):  {unexpected_real}")
 
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device}")
    return model