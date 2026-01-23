import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from model.config import ModelConfig
from model.embedding import EmbeddingLayer
from model.transformer_block import TransformerBlock
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = EmbeddingLayer(config)
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.token_embedding.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights (GPT-2 style)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(
        self, 
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            input_ids: (batch_size, seq_len)
            labels: (batch_size, seq_len) - optional for training
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar (if labels provided)
        """
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            input_ids: (batch_size, seq_len) starting tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling (None = disabled)
            top_p: nucleus sampling threshold
        
        Returns:
            generated_ids: (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            
            # Forward
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat((input_ids, idx_next), dim=1)
        
        return input_ids
if __name__ == "__main__":
    from model.config import ModelConfig
    cfg = ModelConfig()
    model = DecoderOnlyTransformer(cfg)
    # Test forward
    dummy = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(dummy, labels=dummy)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    generated = model.generate(dummy[:1, :5], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")