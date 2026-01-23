import torch
from torch.utils.data import DataLoader
from model import ModelConfig, DecoderOnlyTransformer

def train():
    # Setup
    config = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DecoderOnlyTransformer(config).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate
    )
    
    # Training loop
    for epoch in range(config.epochs):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            logits, loss = model(input_ids, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')

if __name__ == "__main__":
    train()