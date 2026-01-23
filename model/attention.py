import math
import torch
import torch.nn as nn
from model.config import ModelConfig

# Multi-head self-attention
class MaskedSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        )

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, d_model)
        B, T, C = x.size()

        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to multi-head
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (B, n_heads, T, T)

        # apply causal mask
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, n_heads, T, head_dim)

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

if __name__ == "__main__":
    from model.config import ModelConfig

    cfg = ModelConfig()
    attn = MaskedSelfAttention(cfg)

    dummy = torch.randn(2, 5, cfg.d_model)
    out = attn(dummy)

    print(out.shape)
