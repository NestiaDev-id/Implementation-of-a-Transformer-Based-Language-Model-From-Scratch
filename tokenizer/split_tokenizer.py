import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- KONFIGURASI PATH ---
BASE_DIR = Path("D:/FileCoding/llm_scratch")
INPUT_BIN = BASE_DIR / "data/tokenized/tokenized_train_128k.bin"
OUTPUT_DIR = BASE_DIR / "data/tokenized/tokenized_train_chunks"
# 5M Records (Asumsi per record diakhiri token EOS ID 3 seperti script sebelumnya)
CHUNK_SIZE = 5_000_000 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_raw_bin_efficiently(input_path, output_dir, chunk_size):
    if not input_path.exists():
        print(f"❌ File tidak ditemukan: {input_path}")
        return

    # Hitung total elemen uint32 dalam file
    file_size = os.path.getsize(input_path)
    total_elements = file_size // 4 # karena uint32 = 4 byte
    
    print(f"📖 Memproses file sebesar {file_size / (1024**3):.2f} GB")
    print(f"🔢 Estimasi total token: {total_elements:,}")

    # Gunakan Memory Mapping agar tidak memakan RAM meskipun file > 100GB
    data = np.memmap(input_path, dtype=np.uint32, mode='r')
    
    # Mencari posisi EOS (End Of Sentence). 
    # Jika script sebelumnya menggunakan ids.append(3), maka token 3 adalah pemisah.
    EOS_TOKEN = 3
    
    # Temukan indeks semua EOS token
    print("🔍 Mencari batas baris (EOS tokens)...")
    eos_indices = np.where(data == EOS_TOKEN)[0]
    total_records = len(eos_indices)
    print(f"📝 Total ditemukan {total_records:,} baris.")

    chunk_idx = 0
    start_pos = 0

    # Lakukan Looping per CHUNK_SIZE (misal per 5 juta baris)
    for i in range(0, total_records, chunk_size):
        end_record_idx = min(i + chunk_size, total_records)
        
        # Ambil posisi indeks terakhir untuk chunk ini
        actual_end_pos = eos_indices[end_record_idx - 1] + 1
        
        # Tentukan nama file
        start_m = i // 1_000_000
        end_m = end_record_idx // 1_000_000
        out_name = f"chunk_{chunk_idx:03d}_{start_m}M_{end_m}M.bin"
        out_path = Path(output_dir) / out_name
        
        # Tulis data biner
        print(f"📦 Menulis {out_name}...")
        chunk_data = data[start_pos:actual_end_pos]
        chunk_data.tofile(out_path)
        
        start_pos = actual_end_pos
        chunk_idx += 1

    print(f"\n✅ BERHASIL! Data telah dipecah menjadi {chunk_idx} chunk.")

if __name__ == "__main__":
    split_raw_bin_efficiently(INPUT_BIN, OUTPUT_DIR, CHUNK_SIZE)
