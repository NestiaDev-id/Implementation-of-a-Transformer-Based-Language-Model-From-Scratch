import torch
import torch.nn as nn
from model.config import ModelConfig
from model.attention import MaskedSelfAttention

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MaskedSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x):
        # Pre-LN attention
        x = x + self.attn(self.ln1(x))

        # Pre-LN FFN
        x = x + self.ffn(self.ln2(x))

        return x

if __name__ == "__main__":
    from model.config import ModelConfig

    cfg = ModelConfig()
    block = TransformerBlock(cfg)

    dummy = torch.randn(2, 8, cfg.d_model)
    out = block(dummy)

    print(out.shape)
