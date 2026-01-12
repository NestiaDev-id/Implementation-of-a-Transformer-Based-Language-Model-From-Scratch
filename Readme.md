# LLM From Scratch (Transformer Implementation)

Proyek ini adalah implementasi **End-to-End Large Language Model (LLM) Training Pipeline** yang dibangun dari nol (from scratch) menggunakan PyTorch. Proyek ini dirancang untuk tujuan edukasi dan penelitian, dengan fokus pada pemahaman mendalam mengenai arsitektur **Transformer** dan proses pelatihan model bahasa.

Implementasi ini mengacu pada makalah seminal _"Attention Is All You Need"_ (Vaswani et al., 2017) namun disederhanakan menjadi model _Decoder-only_ (seperti GPT) untuk tugas _Causal Language Modeling_.

---

## 📂 Struktur Proyek

Struktur folder disusun secara modular untuk memisahkan antara data, arsitektur model, dan logika pelatihan.

```text
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
├── model/               # Arsitektur Neural Network (Transformer)
│   ├── embedding.py     # Token Embedding & Positional Encoding
│   ├── attention.py     # Multi-Head Self-Attention
│   ├── transformer.py   # Blok Decoder & Model Utama
│   └── lm_head.py       # Output Layer
│
├── training/            # Logika Pelatihan (Training Loop)
│   ├── dataset.py       # PyTorch Dataset & DataLoader
│   ├── train.py         # Script utama pelatihan
│   └── config.yaml      # Konfigurasi Hyperparameter
│
├── scripts/             # Utilitas Pengolahan Data
│   └── preprocess_data.py
│
├── inference/           # Pengujian Model
│   └── generate.py      # Script untuk generate teks
│
└── requirements.txt     # Daftar dependensi
```

---

## 🚀 Cara Menjalankan (Pipeline)

Ikuti langkah-langkah berikut secara berurutan untuk melatih model dari data mentah hingga siap digunakan.

### 0. Persiapan Lingkungan

Pastikan Python sudah terinstal, lalu instal library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

### 1. Data Preparation (Stage 1)

Letakkan file dataset mentah Anda (misalnya `raw_corpus.txt`) di dalam folder `data/raw/`. Kemudian jalankan script cleaning:

```bash
python scripts/preprocess_data.py
```

_Output: `data/cleaned/cleaned_corpus.txt`_

### 2. Tokenizer Construction (Stage 2)

Latih tokenizer (BPE/Byte-Pair Encoding) menggunakan data yang sudah dibersihkan untuk membuat vocabulary.

```bash
python tokenizer/train_tokenizer.py
```

_Output: `tokenizer/tokenizer.model` dan `tokenizer/tokenizer.vocab`_

### 3. Training Model (Stage 3 & 4)

Mulai proses pelatihan. Anda dapat mengatur hyperparameter (seperti `batch_size`, `learning_rate`, `n_layer`) di file `training/config.yaml`.

```bash
python training/train.py
```

_Output: Checkpoint model akan disimpan di folder `checkpoints/`_

### 4. Inference / Text Generation (Stage 5)

Gunakan model yang sudah dilatih untuk menghasilkan teks baru berdasarkan prompt.

```bash
python inference/generate.py --prompt "Artificial Intelligence adalah"
```

---

## 🧠 Metodologi & Arsitektur

Proyek ini mengimplementasikan komponen-komponen kunci berikut:

1.  **Tokenization**: Menggunakan Byte-Pair Encoding (BPE) untuk menangani _out-of-vocabulary words_.
2.  **Embedding**: Input embedding ditambah dengan _Positional Encoding_ (sinusoidal).
3.  **Self-Attention**: Mekanisme _Scaled Dot-Product Attention_ dengan _Causal Masking_ (agar model tidak melihat masa depan).
4.  **Feed-Forward Network**: Dua layer linear dengan aktivasi ReLU/GELU.
5.  **Normalization**: Layer Normalization (Pre-Norm atau Post-Norm).

---

## 📝 Catatan

- Pastikan Anda memiliki memori (RAM/VRAM) yang cukup jika melatih dengan dataset besar.
- Gunakan GPU (CUDA) untuk mempercepat proses pelatihan. Konfigurasi device diatur otomatis di dalam script.

---
