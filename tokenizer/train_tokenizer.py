import sentencepiece as spm
import numpy as np
import random  # Perlu diimpor
from tqdm import tqdm
from pathlib import Path

# --- KONFIGURASI ---
INPUT_DIR = Path("D:/data/cleaned/")
MODEL_PREFIX = "tokenizer/multilingual_128k"
VOCAB_SIZE = 128256
TOKENIZER_CORPUS_FILE = "data/tokenized/tokenizer_corpus_combined_128k.txt" # File gabungan sementara
OUTPUT_BIN = "data//tokenized/tokenized_train_128k.bin"
TOTAL_SAMPLE_LINES = 10_000_000 

# Pastikan direktori ada
Path("tokenizer").mkdir(exist_ok=True)
Path("data/tokenized").mkdir(exist_ok=True)


def reservoir_sample(filepath, sample_size):
    """Fungsi sampling yang efisien untuk file besar."""
    sample = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < sample_size:
                sample.append(line)
            elif random.random() < sample_size / (i + 1):
                replace_index = random.choice(range(sample_size))
                sample[replace_index] = line
    return sample

# Bobot Jatah (Harus total 1.0)
WEIGHTS = {
    'processed_indo.txt': 0.5,   # 50% jatah
    'processed_fr_en.txt': 0.3,  # 30% jatah
    'processed_de_en.txt': 0.2   # 20% jatah (Meskipun file kecil, kita boost kehadirannya)
}

def create_weighted_corpus():
    all_lines = []
    
    for filename, weight in WEIGHTS.items():
        filepath = INPUT_DIR / filename
        if not filepath.exists():
            print(f"⚠️ Skip {filename}, file tidak ditemukan.")
            continue
            
        target_count = int(TOTAL_SAMPLE_LINES * weight)
        print(f"⏳ Mengambil {target_count:,} baris dari {filename} (Weight: {weight*100}%)...")
        
        # Gunakan sampling acak sederhana atau baca baris pertama
        with open(filepath, "r", encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line)
                if i >= target_count * 2: # Baca sedikit lebih banyak untuk di-shuffle nanti
                    break
            
            # Ambil acak sesuai target_count
            sampled = random.sample(lines, min(len(lines), target_count))
            all_lines.extend(sampled)

    if not all_lines:
        return False

    random.shuffle(all_lines)
    with open(TOKENIZER_CORPUS_FILE, "w", encoding="utf-8") as f:
        f.writelines(all_lines)
    return True

# --- LANGKAH 1: TRAINING TOKENIZER ---
if not Path(f"{MODEL_PREFIX}.model").exists():
    if create_weighted_corpus():
        print("🚀 Melatih SentencePiece dengan Weighted Sampling...")
        spm.SentencePieceTrainer.train(
            input=TOKENIZER_CORPUS_FILE,
            model_prefix=MODEL_PREFIX,
            vocab_size=VOCAB_SIZE,
            model_type="bpe",
            character_coverage=0.9995,
            split_digits=True,
            byte_fallback=True,
            user_defined_symbols='[ID],[DE],[EN],[FR]',
            )
    print("✅ Tokenizer Model berhasil dibuat.")
else:
    print("ℹ️ Tokenizer Model sudah ada, melewati Langkah 1.")


# --- LANGKAH 2: TOKENIZING DATASET ---
DATA_SOURCES = [
    "D:/data/cleaned/processed_indo.txt",
    "D:/data/cleaned/processed_de_en.txt",
    "D:/data/cleaned/processed_fr_en.txt"
]

print("⏳ Langkah 2: Mengonversi SEMUA Dataset ke Biner (Tokenizing)...")
sp = spm.SentencePieceProcessor(model_file=f"{MODEL_PREFIX}.model")

# Gunakan mode 'ab' (append binary) agar tidak memakan RAM
with open(OUTPUT_BIN, "wb") as bin_file:
    for file_path in DATA_SOURCES:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️ File tidak ditemukan: {file_path}")
            continue
            
        print(f"Processing: {path.name}...")
        
        # Baca baris demi baris (Streaming)
        with open(path, "r", encoding="utf-8") as f:
            # Batching kecil untuk mempercepat penulisan ke disk
            batch_ids = []
            for i, line in enumerate(tqdm(f)):
                # 1. Encode teks ke ID
                ids = sp.encode_as_ids(line)
                # 2. Tambah EOS (End of Sentence) ID 3
                ids.append(3) 
                batch_ids.extend(ids)
                
                # Setiap 100.000 baris, tulis ke disk dan kosongkan list di RAM
                if i % 100000 == 0 and i > 0:
                    ids_array = np.array(batch_ids, dtype=np.uint32)
                    bin_file.write(ids_array.tobytes())
                    batch_ids = []
            
            # Tulis sisa batch terakhir untuk file ini
            if batch_ids:
                ids_array = np.array(batch_ids, dtype=np.uint32) # menggunakan uint32 jika voca lebih besar
                bin_file.write(ids_array.tobytes())

print(f"✨ SEMUA DATA (30GB+) BERHASIL DITOKENISASI!")
print(f"💾 File biner siap untuk training: {OUTPUT_BIN}")