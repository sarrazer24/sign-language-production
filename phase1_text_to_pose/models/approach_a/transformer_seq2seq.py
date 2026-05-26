
# Approche A — T5-small Encoder + Transformer Decoder
# Nafissa  (Encoder) + Serine  (Decoder)


import torch
import torch.nn as nn
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer



# 1. T5 Encoder  

class T5Encoder(nn.Module):
    """
    Encodeur T5-small fine-tuné partiellement.
    - Couches 0-3 : gelées 
    - Couches 4-5 : fine-tunées 
    Input  : textes (list of str)
    Output : memory (B, T_text, 512)
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.encoder   = T5EncoderModel.from_pretrained("t5-small")

        #  geler tout
        for param in self.encoder.parameters():
            param.requires_grad = False

        #  ouvrir les 2 dernières couches
        for param in self.encoder.encoder.block[-2:].parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f" T5Encoder | trainable: {trainable:,} / {total:,}")

    def forward(self, texts, device):
        tokens = self.tokenizer(
            texts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = 128
        ).to(device)

        out = self.encoder(
            input_ids      = tokens["input_ids"],
            attention_mask = tokens["attention_mask"]
        )
        return out.last_hidden_state  # (B, T_text, 512)



# 2. Pose Decoder  

class PoseDecoder(nn.Module):
    """
    Décodeur Transformer autorégressif.
    - 4 couches TransformerDecoder
    - Cross-Attention avec memory T5
    - Causal mask pour ne pas voir le futur
    Input  : poses (B, T, pose_dim) + memory (B, T_text, 512)
    Output : pred  (B, T, pose_dim)
    """
    def __init__(self, pose_dim=453, d_model=256):
        super().__init__()
        self.pose_embed  = nn.Linear(pose_dim, d_model)
        self.memory_proj = nn.Linear(512, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model         = d_model,
            nhead           = 8,
            dim_feedforward = 1024,
            dropout         = 0.1,
            batch_first     = True
        )
        self.decoder    = nn.TransformerDecoder(dec_layer, num_layers=4)
        self.projection = nn.Linear(d_model, pose_dim)

        print(f" PoseDecoder | pose_dim={pose_dim} | d_model={d_model}")

    def forward(self, poses, memory, pose_mask):
        B, T, _ = poses.shape

        # Causal mask — ne voit pas le futur
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=poses.device
        )

        # Padding mask
        pad_mask = ~pose_mask.bool()

        # Projections
        memory = self.memory_proj(memory)  # 512  → 256
        x      = self.pose_embed(poses)    # 453  → 256

        out = self.decoder(
            tgt                  = x,
            memory               = memory,
            tgt_mask             = causal_mask,
            tgt_key_padding_mask = pad_mask
        )
        return self.projection(out)  # 256 → 453



# 3. Loss

def scale_adjusted_loss(pred, target, mask, lengths=None):
    """
    MSE Loss sur les frames réelles uniquement (ignore padding).
    pred   : (B, T, D)
    target : (B, T, D)
    mask   : (B, T) — 1=réel, 0=padding
    """
    loss = (pred - target) ** 2        # (B, T, D)
    loss = loss.mean(dim=-1)           # (B, T)
    loss = loss * mask.float()         # ignore padding
    loss = loss.sum() / (mask.float().sum() + 1e-8)
    return loss



# 4. SignModel — Modèle Complet

class SignModel(nn.Module):
    """
    Modèle complet Approche A:
    texts + poses → pred poses

    Pipeline:
    texts → T5Encoder → memory (B, T_text, 512)
    poses + memory → PoseDecoder → pred (B, T, 453)
    """
    def __init__(self, pose_dim=453):
        super().__init__()
        self.encoder = T5Encoder()
        self.decoder = PoseDecoder(pose_dim=pose_dim)

    def forward(self, texts, poses, pose_mask, device):
        memory = self.encoder(texts, device)
        pred   = self.decoder(poses, memory, pose_mask)
        return pred

    def generate(self, texts, max_frames=200, device='cuda'):
        """
        Génération autoregresssive:
        texts → poses frame par frame
        sans poses de référence
        """
        self.eval()
        B = len(texts)

        with torch.no_grad():
            # Encoder le texte
            memory = self.encoder(texts, device)  # (B, T_text, 512)

            # Premier frame = zéros
            pose_dim = self.decoder.projection.out_features
            poses    = torch.zeros(B, 1, pose_dim).to(device)
            mask     = torch.ones(B, 1).to(device)

            for t in range(max_frames - 1):
                # Prédire le prochain frame
                pred = self.decoder(poses, memory, mask)

                # Prendre le dernier frame prédit
                next_frame = pred[:, -1:, :]  # (B, 1, 453)

                # Ajouter au séquence
                poses = torch.cat([poses, next_frame], dim=1)
                mask  = torch.ones(B, poses.shape[1]).to(device)

        return poses  # (B, max_frames, 453)


