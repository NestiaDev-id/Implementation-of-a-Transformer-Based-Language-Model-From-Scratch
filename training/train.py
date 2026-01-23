import torch
from torch.utils.data import DataLoader
from model.model import DecoderOnlyTransformer
from model.config import ModelConfig
from training.dataset import TextDataset
import os

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("checkpoints", exist_ok=True)

    config = ModelConfig()
    model = DecoderOnlyTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # scaler untuk mempercepat training (Mixed Precision)
    scaler = torch.amp.GradScaler('cuda')

    # Gunakan path file bin/npy, bukan torch.load
    dataset = TextDataset("data/tokenized_train.bin", config.max_seq_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    model.train()
    print(f"🚀 Training started on {device}...")

    for epoch in range(config.epochs):
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True) # Lebih cepat dari zero_grad() biasa

            # Mixed Precision Training
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x) # Output: [B, T, V]
                
                # Reshape untuk CrossEntropy
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    y.view(-1)
                )

            # Backward pass dengan scaler
            scaler.scale(loss).backward()
            
            # Gradient Clipping (Penting agar model tidak 'meledak')
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()

            if step % 100 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

        # Simpan Checkpoint setiap akhir epoch
        checkpoint_path = f"checkpoints/llm_epoch_{epoch}.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config
        }, checkpoint_path)
        print(f"✅ Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    train()
