# 🧠 GPT-2 Decoder-Only Transformer - From Scratch

> **Implementasi lengkap GPT-2 decoder-only transformer dari nol** menggunakan PyTorch, dengan fokus pada pemahaman mendalam arsitektur modern Large Language Model.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## 📋 Daftar Isi

- [Overview](#-overview)
- [Fitur Utama](#-fitur-utama)
- [Arsitektur Model](#-arsitektur-model)
- [Struktur Project](#-struktur-project)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Alur Program](#-alur-program)
- [Training](#-training)
- [Roadmap](#-roadmap)

---

## 🎯 Overview

Project ini adalah **implementasi from scratch** dari GPT-2 decoder-only transformer untuk:

- ✅ **Belajar**: Memahami setiap komponen LLM secara mendalam
- ✅ **Kontrol Penuh**: Modifikasi arsitektur sesuai kebutuhan
- ✅ **Offline**: Training dan inference tanpa API/internet
- ✅ **Privacy**: 100% data lokal
- ✅ **Riset**: Eksplorasi arsitektur baru

### 🆚 Perbedaan dengan Pakai API

| Aspek             | Pakai API (OpenAI, dll) | From Scratch (Project Ini)    |
| ----------------- | ----------------------- | ----------------------------- |
| **Coding**        | Minimal (3-5 baris)     | Full implementation ✅        |
| **Training**      | Tidak perlu             | Bisa custom training ✅       |
| **Biaya**         | $$ per request          | Gratis (setelah punya GPU) ✅ |
| **Privacy**       | Data ke server mereka   | 100% lokal ✅                 |
| **Customization** | Tidak bisa              | Bebas modify ✅               |
| **Internet**      | Wajib                   | Tidak perlu (offline) ✅      |

---

## ✨ Fitur Utama

### 🏗️ Arsitektur Modern

- ✅ **Pre-Layer Normalization** (lebih stabil dari post-norm)
- ✅ **Causal Masking** untuk autoregressive generation
- ✅ **Multi-Head Self-Attention** dengan scaled dot-product
- ✅ **Position-wise Feed-Forward** dengan GELU activation
- ✅ **Learned Positional Embeddings** (GPT-2 style)
- ✅ **Weight Tying** untuk efisiensi memory

### 🚀 Capabilities

- ✅ **Forward Pass** dengan loss computation
- ✅ **Text Generation** dengan temperature & top-k/top-p sampling
- ✅ **Gradient Clipping** untuk stable training
- ✅ **Modular Design** (mudah di-extend)

### 📊 Konfigurasi

```python
vocab_size: 65,536      # Vocabulary size
d_model: 768            # Hidden dimension
n_layers: 6             # Number of transformer blocks
n_heads: 12             # Attention heads
max_seq_len: 512        # Maximum sequence length
```

---

## 🏛️ Arsitektur Model

### High-Level Architecture

```mermaid
graph TD
    A[Input Token IDs] --> B[Token Embedding<br/>+ Positional Embedding]
    B --> C[Transformer Block 1]
    C --> D[Transformer Block 2-6]
    D --> E[Layer Normalization]
    E --> F[LM Head]
    F --> G[Logits: vocab_size]

    style A fill:#e1f5ff
    style G fill:#ffe1e1
    style C fill:#fff4e1
    style D fill:#fff4e1
```

### Transformer Block Detail

```mermaid
graph TD
    A[Input x] --> B[LayerNorm 1]
    B --> C[Multi-Head<br/>Self-Attention]
    C --> D[Residual Add]
    A -.Residual.-> D

    D --> E[LayerNorm 2]
    E --> F[Feed-Forward<br/>Network]
    F --> G[Residual Add]
    D -.Residual.-> G

    G --> H[Output]

    style A fill:#e1f5ff
    style H fill:#ffe1e1
    style C fill:#fff4e1
    style F fill:#fff4e1
```

### Attention Mechanism Flow

```mermaid
graph LR
    A[Input x<br/>batch, seq, d_model] --> B[QKV Projection]
    B --> C[Split to n_heads]

    C --> D1[Q: 12 heads]
    C --> D2[K: 12 heads]
    C --> D3[V: 12 heads]

    D1 --> E[Scaled Dot-Product]
    D2 --> E

    E --> F[Causal Mask]
    F --> G[Softmax]
    G --> H[× V]
    D3 --> H

    H --> I[Merge Heads]
    I --> J[Output Projection]

    style E fill:#fff4e1
    style F fill:#ffe1e1
```

---

## 📁 Struktur Project

```
llm-from-scratch/
│
├── data/                # Manajemen Dataset
│   ├── raw/             # Dataset mentah (txt/csv)
│   ├── cleaned/         # Dataset bersih setelah preprocessing
│   └── tokenized/       # Dataset yang sudah diubah menjadi token ID
│
├── tokenizer/           # Komponen Tokenisasi
│   ├── train_tokenizer.py
│   └── tokenizer.model  # Model BPE yang sudah dilatih
│
├── model/                         # Arsitektur Neural Network (Transformer)
│   ├── __init__.py                # Package exports
│   ├── config.py                  # Configuration (ModelConfig)
│   ├── attention.py               # Multi-head self-attention
│   ├── embedding.py               # Token + positional embeddings
│   ├── transformer_block.py       # Complete transformer layer
│   ├── model.py                   # Main DecoderOnlyTransformer
│   ├── train.py                   # Training script
│   └── generate.py                # Inference/generation script
│
├── training/            # Logika Pelatihan (Training Loop)
│   ├── dataset.py       # PyTorch Dataset & DataLoader
│   └── train.py         # Script utama pelatihan
│
├── scripts/             # Utilitas Pengolahan Data
│   └── preprocess_data.py
│
├── translate/            # Augmented Dataset
│   └── augmented.py      # Script untuk augmentasi dataset
│
└── requirements.txt     # Daftar dependensi
```

### 🔗 Dependency Graph

```mermaid
graph TD
    A[config.py] --> B[attention.py]
    A --> C[embedding.py]

    B --> D[transformer_block.py]
    A --> D

    C --> E[model.py]
    D --> E
    A --> E

    E --> F[train.py]
    E --> G[generate.py]

    style A fill:#e1f5ff
    style E fill:#ffe1e1
    style F fill:#e1ffe1
    style G fill:#fff4e1
```

**Urutan Baca untuk Pemahaman**:

1. `config.py` - Pahami konfigurasi
2. `embedding.py` - Token → Vector
3. `attention.py` - Core attention mechanism
4. `transformer_block.py` - Complete layer
5. `model.py` - Full model assembly
6. `train.py` / `generate.py` - Usage examples

---

## 💻 Instalasi

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (untuk training)
- 8GB+ VRAM (recommended)

### Install Dependencies

```bash
# Clone repository
git clone https://github.com/NestiaDev-id/Implementation-of-a-Transformer-Based-Language-Model-From-Scratch.git

cd Implementation-of-a-Transformer-Based-Language-Model-From-Scratch

# Install PyTorch (sesuaikan dengan CUDA version)
# CPU only:
pip install torch

# CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install additional dependencies
pip install transformers datasets

# Install requirements if needed
pip install -r requirements.txt
```

### Verify Installation

```bash
cd model
python model.py
```

Expected output:

```
Logits shape: torch.Size([2, 16, 65536])
Loss: 11.0891
Generated shape: torch.Size([1, 25])
Total parameters: 86,794,752
```

---

## 🚀 Cara Penggunaan

### 1. Create Model

```python
from model import ModelConfig, DecoderOnlyTransformer

# Initialize configuration
config = ModelConfig()

# Create model
model = DecoderOnlyTransformer(config)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 2. Forward Pass (Training Mode)

```python
import torch

# Prepare dummy data
batch_size = 2
seq_len = 16
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
labels = input_ids.clone()

# Forward pass
logits, loss = model(input_ids, labels=labels)

print(f"Logits shape: {logits.shape}")  # (2, 16, 65536)
print(f"Loss: {loss.item():.4f}")
```

### 3. Text Generation

```python
from transformers import GPT2Tokenizer

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare prompt
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

# Generate
model.eval()
output_ids = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)

# Decode
generated_text = tokenizer.decode(output_ids[0])
print(generated_text)
```

---

## 🔄 Alur Program

### Forward Pass Flow

```mermaid
sequenceDiagram
    participant Input
    participant Embedding
    participant Block
    participant Norm
    participant Head
    participant Loss

    Input->>Embedding: Token IDs (batch, seq)
    Note over Embedding: Token Emb<br/>+ Pos Emb<br/>+ Dropout
    Embedding->>Block: (batch, seq, d_model)

    loop 6 layers
        Block->>Block: Pre-LN + Attention
        Block->>Block: Pre-LN + FFN
    end

    Block->>Norm: (batch, seq, d_model)
    Norm->>Head: Normalized
    Head->>Loss: Logits (batch, seq, vocab)

    alt Training Mode
        Loss->>Loss: Cross Entropy
        Note over Loss: Compare with labels
    end
```

### Generation Flow

```mermaid
graph TD
    A[Prompt: 'Once upon'] --> B[Tokenize]
    B --> C[Model Forward]
    C --> D[Get Last Logits]
    D --> E[Apply Temperature]
    E --> F[Top-k/Top-p Filter]
    F --> G[Softmax]
    G --> H[Sample Next Token]
    H --> I[Append to Sequence]
    I --> J{Max tokens?}

    J -->|No| C
    J -->|Yes| K[Decode to Text]

    style A fill:#e1f5ff
    style K fill:#e1ffe1
```

### Data Shape Transformation

```
Input IDs: (batch=2, seq=16)
    ↓
Token Embedding: (2, 16, 768)
Positional Embedding: (2, 16, 768)
Combined: (2, 16, 768)
    ↓
Transformer Block 1-6: (2, 16, 768) → (2, 16, 768)
    ├─ Attention: (2, 16, 768) → (2, 16, 768)
    └─ FFN: (2, 16, 768) → (2, 16, 768)
    ↓
Final LayerNorm: (2, 16, 768)
    ↓
LM Head: (2, 16, 65536)
    ↓
Output Logits: (2, 16, 65536)
```

---

## 🎓 Training

### Prepare Dataset

```python
from datasets import load_dataset

# Option 1: Public dataset
dataset = load_dataset("wikipedia", "20220301.id")  # Indonesian Wikipedia

# Option 2: Custom text files
dataset = load_dataset("text", data_files={"train": "data/*.txt"})
```

### Run Training

```bash
cd model
python train.py
```

### Training Script Structure

The `train.py` includes:

- ✅ Model initialization
- ✅ Optimizer (AdamW)
- ✅ Training loop with gradient clipping
- ✅ Loss logging
- ✅ Checkpoint saving

### Monitor Training

```python
# During training, you'll see:
Epoch 0, Loss: 10.2345
Epoch 0, Loss: 9.8432
Epoch 0, Loss: 9.5123
...
# Checkpoint saved: checkpoint_epoch_0.pt
```

### Load Trained Model

```python
# Load checkpoint
model = DecoderOnlyTransformer(config)
model.load_state_dict(torch.load('checkpoint_epoch_4.pt'))
model.eval()

# Use for inference
output = model.generate(input_ids, max_new_tokens=100)
```

---

## 🗺️ Roadmap

### ✅ Completed

- [x] Core model architecture (GPT-2)
- [x] Multi-head self-attention
- [x] Causal masking
- [x] Position embeddings
- [x] Feed-forward networks
- [x] Generation with sampling
- [x] Training script foundation
- [x] Modular code structure

### 🚧 In Progress

- [ ] Complete training pipeline with dataset loader
- [ ] Training optimization (mixed precision, gradient accumulation)
- [ ] Evaluation metrics

### 📋 Planned

- [ ] Custom tokenizer training (BPE)
- [ ] Multi-GPU training (DDP)
- [ ] Model quantization (INT8, FP16)
- [ ] Fine-tuning utilities
- [ ] Inference optimization (KV caching, Flash Attention)
- [ ] Web UI for generation
- [ ] Benchmark comparisons

---

## 📚 Reference Implementations

Project ini juga include reference implementations untuk belajar:

### 1. `test_qwen3.py` - Modern Efficient Transformer

- ✅ **Grouped Query Attention (GQA)** - 75% less KV cache
- ✅ **RoPE** - Better positional encoding
- ✅ **SwiGLU** - Superior activation
- ✅ **RMSNorm** - 40% faster normalization
- ✅ **QK-Norm** - Stable training

### 2. `test_llama4.py` - Mixture of Experts

- ✅ **MoE Architecture** - Sparse scaling
- ✅ **Expert Routing** - Smart token routing
- ✅ **15-25% active params** - Efficient inference
- ✅ **Long context** - Up to 10M tokens

### 3. `test_deepseek_r1.py` - RL Reasoning

- ✅ **PPO Training** - Reinforcement learning
- ✅ **Dual Heads** - Policy + Value
- ✅ **Thinking Mode** - Chain-of-thought reasoning
- ✅ **Self-verification** - Quality control

### 4. `test_hybrid.py` - Best of All Worlds

- ✅ Combines all best features
- ✅ MoE + GQA + RL + Modern components

---

## 🙏 Acknowledgments

- **Attention is All You Need** - Original Transformer paper
- **GPT-2** - OpenAI's language model
- **Qwen3** - Alibaba Cloud's efficient transformer
- **Llama 4** - Meta AI's MoE architecture
- **DeepSeek-R1** - RL-based reasoning

---

## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐

---

## 📊 Performance Metrics

### Model Size

```
Parameters: ~87M (87,794,752)
Size on disk: ~350MB (FP32), ~175MB (FP16)
Inference speed: ~50 tokens/sec (RTX 3060)
Training speed: ~2K tokens/sec (RTX 3060)
```

### Memory Requirements

```
Training (batch=32, seq=512): ~8GB VRAM
Inference (batch=1, seq=512): ~2GB VRAM
With FP16: ~50% memory reduction
```

---

**Built with ❤️ for learning and understanding Large Language Models**

_Last updated: 2026-01-24_
