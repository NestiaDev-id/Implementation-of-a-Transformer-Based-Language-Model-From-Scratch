from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 65.536       # Kamus kata, sesuai hasil training tokenizer sebelumnya. Standar 50.257 untuk GPT-2, 65.536 untuk penelitian ini
    
    # Sequence
    max_seq_len: int = 512       # Berapa banyak kata yang diingat AI dalam satu waktu
    
    # Model dimension
    d_model: int = 768            # Dimensi vektor (Hidden Size), 768 standar untuk GPT-2
    n_layers: int = 6            # Jumlah blok Transformer (kedalaman otak), 6 standar, 12 untuk GPT-3, 24 untuk PaLM, 32 untuk Llama, 40 untuk ChatGLM, 64 untuk Llama-2, 80 untuk ChatGLM-6B
    n_heads: int = 8             # Jumlah attention heads (d_model harus habis dibagi n_heads)
    d_ff: int = 2048            # Dimensi feed-forward layer (2.3-4x d_model biasanya) ini adalah jumlah dari d_model * 4

    dropout: float = 0.1          # Mencegah model menghafal (overfitting)
    bias: bool = False            # Standar model modern llm untuk stabilitas
    
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    # Dropout
    dropout: float = 0.1

    # Training-related
    tie_embeddings: bool = True
    
    # Optimizer
    learning_rate: float = 3e-4    # Standar learning ratekece untuk model seukuran ini
    batch_size: int = 32           # Sesuaikan dengan RAM/VRAM Anda
    epochs: int = 5                # Berapa kali mengulang dataset
    
    # Reasoning
    max_reasoning_steps: int = 10
    verification_threshold: float = 0.8
    temperature: float = 0.6            # Semakin tinggi, semakin random atau semakin kreatif. Semakin rendah, semakin stabil atau semakin focus pada jawaban
    top_p: float = 0.95                 
    
    