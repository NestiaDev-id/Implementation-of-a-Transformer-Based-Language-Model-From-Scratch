import hashlib
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm
import ftfy
import pandas as pd
import hashlib
import csv
import random

# --- Konfigurasi Jalur (Path) ---
# Lokasi DATA ASLI (Drive E) - Tidak perlu dipindah/copy
RAW_DATA_SOURCE = Path("E:/Dataset/Wmt14/de-en") 

# Lokasi OUTPUT (Hanya hasil cleaning yang disimpan di folder proyek)
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_FILE = BASE_DIR / "data" / "cleaned" / "deen_clean.txt"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
TOKENIZER_OUTPUT = CLEANED_DIR / "sample_combined_for_tokenizer.txt"

MIN_LEN = 20
MAX_LEN = 2000

def clean_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = unicodedata.normalize('NFKC', text)
    
    # Hapus HTML & URL
    text = re.sub(r"<.*?>", " ", text) 
    text = re.sub(r'http\S+|www\S+', ' ', text)
    
    # Simpan aksen: gunakan lowercase tapi jaga karakter unicode
    text = text.lower() 
    
    # \w mencakup a-z, A-Z, 0-9, dan karakter beraksen seperti ü, é, ß
    # text = re.sub(r"[^a-z0-9.,!?\- ]", " ", text)
    # text = re.sub(r"[^\w\s.,!?\-]", " ", text)
    text = re.sub(r"[^\w\s.,!?\-]", " ", text).replace("_", " ")

    text = re.sub(r"\s+", " ", text)
    return text.strip()

def process_files():
    # Pastikan folder output di proyek ada
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    lines_written = 0

    # Mencari semua file .txt secara rekursif di Drive E
    # Gunakan .rglob("*.txt") jika ada di dalam subfolder
    source_files = list(RAW_DATA_SOURCE.glob("*.txt"))
    if not source_files:
        print(f"❌ Tidak ditemukan file .txt di {RAW_DATA_SOURCE}")
        return

    print(f"🚀 Memproses langsung dari: {RAW_DATA_SOURCE}")
    print(f"📦 Output akan disimpan ke: {OUT_FILE}")

    with OUT_FILE.open("w", encoding="utf-8") as out_f:
        for file_path in source_files:
            # Menggunakan errors='ignore' agar tidak berhenti jika ada karakter rusak di dataset asli
            with file_path.open("r", encoding="utf-8", errors="ignore") as in_f:
                for line in tqdm(in_f, desc=f"Reading {file_path.name}", unit=" lines"):
                    cleaned_line = clean_text(line)
                    
                    if MIN_LEN <= len(cleaned_line) <= MAX_LEN:
                        # Deduplikasi menggunakan binary hash (lebih hemat RAM)
                        line_hash = hashlib.md5(cleaned_line.encode()).digest()
                        
                        if line_hash not in seen_hashes:
                            out_f.write(cleaned_line + "\n")
                            seen_hashes.add(line_hash)
                            lines_written += 1

    print(f"\n✅ Selesai! {lines_written} baris unik berhasil diekstrak.")



def process_files_wmt14():
    datasets = {
        "test": RAW_DATA_SOURCE / "wmt14_translate_de-en_test.csv",
        "train": RAW_DATA_SOURCE / "wmt14_translate_de-en_train.csv",
        "validation": RAW_DATA_SOURCE / "wmt14_translate_de-en_validation.csv"
    }
    
    for split, file_path in datasets.items():
        if not file_path.exists(): continue

        out_split_file = BASE_DIR / "data" / "cleaned" / f"wmt14_deen_{split}_clean.txt"
        out_split_file.parent.mkdir(parents=True, exist_ok=True)
        
        seen_hashes = set()
        lines_written = 0

        print(f"🚀 Memproses {split.upper()}: {file_path.name}")

        with out_split_file.open("w", encoding="utf-8") as out_f:
            reader = pd.read_csv(
                file_path, 
                chunksize=10000, 
                encoding='utf-8', 
                on_bad_lines='skip', 
                quoting=csv.QUOTE_MINIMAL,
                engine='python' # Gunakan Python engine yang lebih lambat tapi kuat
            )

            for chunk in reader:
                for _, row in chunk.iterrows():
                    try:
                        de_raw = str(row['de'])
                        en_raw = str(row['en'])

                        cleaned_de = clean_text(de_raw)
                        cleaned_en = clean_text(en_raw)
                        
                        if len(cleaned_de) >= MIN_LEN and len(cleaned_en) >= MIN_LEN:
                            # Masukkan sebagai dua baris terpisah agar model belajar bahasa murni
                            for cleaned_line in [cleaned_de, cleaned_en]:
                                line_hash = hashlib.md5(cleaned_line.encode()).digest()
                                if line_hash not in seen_hashes:
                                    out_f.write(cleaned_line + "\n")
                                    seen_hashes.add(line_hash)
                                    lines_written += 1
                    except Exception:
                        continue 

        print(f"✅ Selesai {split}: {lines_written} baris unik ke {out_split_file.name}")
    
    
def reservoir_sample(file_path, sample_size):
    """Algoritma Reservoir Sampling untuk mengambil sampel dari file besar tanpa memuat semua ke RAM."""
    reservoir = []
    if not Path(file_path).exists():
        print(f"⚠️ File tidak ditemukan untuk sampling: {file_path}")
        return reservoir

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < sample_size:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = line
    return reservoir

def create_tokenizer_corpus(sample_size_per_file=1_000_000):
    sources = {
        "indo": CLEANED_DIR / "corpus_cleaned.txt",
        "wmt_test": CLEANED_DIR / "wmt14_deen_test_clean.txt",
        "wmt_train": CLEANED_DIR / "wmt14_deen_train_clean.txt",
        "wmt_val": CLEANED_DIR / "wmt14_deen_validation_clean.txt"
    }

    all_lines = []
    print(f"\n--- Memulai Sampling (Target: {sample_size_per_file:,} baris per file) ---")

    for name, path in sources.items():
        if path.exists():
            print(f"Sampling dari {name}...")
            lines = reservoir_sample(path, sample_size_per_file)
            all_lines.extend(lines)
    
    if not all_lines:
        print("❌ Tidak ada data untuk digabungkan!")
        return

    print("Shuffling data...")
    random.shuffle(all_lines)

    with open(TOKENIZER_OUTPUT, "w", encoding="utf-8") as f:
        f.writelines(all_lines)

    print(f"\n✨ Tokenizer corpus siap: {TOKENIZER_OUTPUT}")
    print(f"📊 Total baris gabungan: {len(all_lines):,}")

    
    
if __name__ == "__main__":
    # proses pemberisihan data indoNLP
    # process_files()
    
    # proses pembersihan data Wmpt14 De-En
    # process_files_wmt14()
    
    create_tokenizer_corpus(sample_size_per_file=1_000_000)


    print("\n📦 Output disimpan di:", OUT_FILE)