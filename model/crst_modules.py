import math
import torch
import torch.nn as nn

from .clip_model import LayerNorm, QuickGELU


class ResolutionConditionedReasoner(nn.Module):
    def __init__(self, embed_dim: int, num_resolutions: int = 4, hidden_dim: int = None,
                 init_gate_logit: float = 2.0):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.resolution_embedding = nn.Embedding(num_resolutions, embed_dim)
        self.token_ln = LayerNorm(embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            QuickGELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.resolution_embedding.weight, std=embed_dim ** -0.5)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, init_gate_logit)

    def forward(self, visual_tokens: torch.Tensor, res_label: torch.Tensor = None):
        if res_label is None:
            res_label = torch.zeros(visual_tokens.size(0), dtype=torch.long, device=visual_tokens.device)
        res_label = res_label.to(device=visual_tokens.device, dtype=torch.long).clamp(0, 3)

        cond = self.resolution_embedding(res_label).to(dtype=visual_tokens.dtype)
        cond = cond.unsqueeze(1).expand(-1, visual_tokens.size(1), -1)
        gate_in = torch.cat([self.token_ln(visual_tokens), cond], dim=-1)
        rho = torch.sigmoid(self.gate(gate_in))
        return visual_tokens * rho, rho


class TextGuidedFeatureRefiner(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_v = LayerNorm(embed_dim)
        self.ln_t = LayerNorm(embed_dim)
        self.ln_out = LayerNorm(embed_dim)
        self.gate = nn.Linear(embed_dim * 2, 1)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, visual_feat: torch.Tensor, text_tokens: torch.Tensor,
                text_key_padding_mask: torch.Tensor = None):
        out_dtype = visual_feat.dtype
        attn_dtype = self.cross_attn.in_proj_weight.dtype
        visual_feat = visual_feat.to(attn_dtype)
        text_tokens = text_tokens.to(attn_dtype)

        q = self.ln_v(visual_feat).unsqueeze(1)
        kv = self.ln_t(text_tokens)
        delta, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        delta = delta.squeeze(1)
        gate = torch.sigmoid(self.gate(torch.cat([visual_feat, delta], dim=-1)))
        refined = self.ln_out(visual_feat + gate * delta)
        return refined.to(out_dtype)
