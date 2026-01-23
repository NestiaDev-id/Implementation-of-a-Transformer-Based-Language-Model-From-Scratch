import math
import torch
import torch.nn as nn
from model.config import ModelConfig

class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id
        )

    def forward(self, input_ids: torch.Tensor):
        # input_ids: (batch_size, seq_len)
        return self.embedding(input_ids)
    
class PositionalEmbedding(nn.Module):
    """Learned positional embeddings (GPT-2 style)"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model
        )
    
    def forward(self, position_ids: torch.Tensor):
        # position_ids: (batch_size, seq_len) or (seq_len,)
        return self.embedding(position_ids)
class EmbeddingLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = TokenEmbedding(config)
        self.positional_embedding = PositionalEmbedding(config)  # Changed!
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, input_ids: torch.Tensor):
        seq_len = input_ids.size(1)
        device = input_ids.device
        
        # Create position IDs
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.positional_embedding(position_ids)
        
        x = tok_emb + pos_emb
        return self.dropout(x)

if __name__ == "__main__":
    cfg = ModelConfig()
    layer = EmbeddingLayer(cfg)

    dummy_input = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(2, 10)
    )

    out = layer(dummy_input)
    print(out.shape)
