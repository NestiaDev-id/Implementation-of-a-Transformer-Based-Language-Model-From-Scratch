import torch
from model import ModelConfig, DecoderOnlyTransformer
def generate_text(prompt, model, tokenizer, max_tokens=100):
    device = next(model.parameters()).device
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_k=50
    )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0])
    return generated_text
if __name__ == "__main__":
    cfg = ModelConfig()
    model = DecoderOnlyTransformer(cfg)
    model.load_state_dict(torch.load('checkpoint.pt'))
    
    prompt = "Once upon a time"
    output = generate_text(prompt, model, tokenizer)
    print(output)