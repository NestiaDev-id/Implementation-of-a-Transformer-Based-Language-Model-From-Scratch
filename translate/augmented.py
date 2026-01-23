import pandas as pd
import random
import re
import csv
import json
import requests
from pathlib import Path
from tqdm import tqdm
import nltk
import ftfy
import unicodedata
import time
import os
import polars as pl

# ==============================================================================
# KONFIGURASI PATH
# ==============================================================================

# 1. Dataset Monolingual Indonesia (File Spesifik)
PATH_INDO_FILE = Path("E:/Dataset/IndoNLP/IndoNLG.txt") 
INPUT_FILE = "E:/Dataset/IndoNLP/IndoNLG.txt"
OUTPUT_FILE = "D:/data/cleaned/indonesian_corpus.txt"
# 2. Dataset Parallel WMT14 (Jerman & Prancis)
PATH_WMT_DE = Path("E:/Dataset/Wmt14/de-en/wmt14_translate_de-en_train.csv")
PATH_WMT_FR = Path("E:/Dataset/Wmt14/fr-en/wmt14_translate_fr-en_train.csv") 

# 3. Output Folder
OUTPUT_DIR = Path("/data/cleaned/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# SETTING HYPERPARAMETER
# ==============================================================================
MIN_LEN = 10
MAX_LEN = 2048
ANCHOR_PROBABILITY = 0.4  # 40% kemungkinan kalimat Inggris disuntik kata Indo
WORD_REPLACE_RATIO = 0.3  # Jika disuntik, 30% kata diganti

# ==============================================================================
# 1. PERSIAPAN KAMUS (MUSE Facebook EN-ID)
# ==============================================================================
print("⏳ Menyiapkan NLTK & Dictionary...")
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('taggers/universal_tagset')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')

def load_comprehensive_dictionary():
    """
    Download & Load kamus Facebook MUSE (EN-ID) secara otomatis.
    """
    local_dict_path = OUTPUT_DIR / "muse_en_id.txt"
    MUSE_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-id.txt"
    final_dict = {}

    # 1. Download jika belum ada
    if not local_dict_path.exists():
        print(f"📥 Mengunduh kamus MUSE EN-ID...")
        try:
            response = requests.get(MUSE_URL)
            response.raise_for_status()
            with open(local_dict_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print("✅ Download selesai!")
        except Exception as e:
            print(f"⚠️ Gagal download: {e}. Menggunakan fallback.")
            return {"house": "rumah", "book": "buku", "love": "cinta", "eat": "makan"}

    # 2. Parsing ke Dictionary
    print(f"📖 Membaca kamus...")
    try:
        with open(local_dict_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    en, id_word = parts[0].strip(), parts[1].strip()
                    if len(en) > 2 and len(id_word) > 2:
                        final_dict[en] = id_word
        print(f"✅ Kamus siap! Total: {len(final_dict):,} kata.")
        return final_dict
    except Exception:
        return {}

ANCHOR_DICT = load_comprehensive_dictionary()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = ftfy.fix_text(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def apply_code_switching(text):
    words = text.split()
    if len(words) < 3: return text
    
    # Gunakan tagset universal (NOUN, VERB, ADJ)
    tagged = nltk.pos_tag(words, tagset='universal')
    new_words = []
    
    for word, tag in tagged:
        clean_word = re.sub(r'[^\w]', '', word).lower()
        if tag in ['NOUN', 'VERB', 'ADJ'] and clean_word in ANCHOR_DICT:
            if random.random() < WORD_REPLACE_RATIO:
                replacement = ANCHOR_DICT[clean_word]
                if word[0].isupper(): replacement = replacement.capitalize()
                new_words.append(replacement)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return " ".join(new_words)

# ==============================================================================
# 2. PROSES DATASET PARALEL (Jerman & Prancis)
# ==============================================================================
def process_parallel_dataset(name, input_path, prefix_token, target_col):
    if not input_path.exists():
        print(f"⚠️ Dataset {name} tidak ditemukan di {input_path}")
        return

    output_file = OUTPUT_DIR / f"processed_{name}.txt"
    file_size = os.path.getsize(input_path) / (1024**3) # Ukuran dalam GB
    
    print(f"\n🚀 Memproses {name.upper()} (Augmented)")
    print(f"📦 Ukuran File: {file_size:.2f} GB")
    
    lines_written = 0
    start_time = time.time()
    
    # Gunakan tqdm untuk melihat progress bar
    # Karena pandas chunk tidak tahu total baris, kita gunakan total estimasi atau biarkan dinamis
    with open(output_file, "w", encoding="utf-8") as f_out:
        # Gunakan chunksize yang lebih besar untuk efisiensi (misal 10000)
        reader = pd.read_csv(input_path, chunksize=10000, encoding='utf-8', 
                             on_bad_lines='skip', engine='python') # 'c' engine lebih cepat dari 'python'
        
        pbar = tqdm(unit=" baris", desc=f"⏳ Processing {name}")
        
        for chunk in reader:
            batch_output = [] # Gunakan list buffer untuk menulis lebih cepat
            for _, row in chunk.iterrows():
                try:
                    src = clean_text(str(row['en']))
                    tgt = clean_text(str(row[target_col]))
                    
                    if len(src) < MIN_LEN or len(tgt) < MIN_LEN: 
                        continue

                    # Data Original
                    batch_output.append(f"{prefix_token} {src} [TARGET] {tgt}\n")
                    lines_written += 1

                    # Data Code-Switching
                    if random.random() < ANCHOR_PROBABILITY:
                        src_aug = apply_code_switching(src)
                        if src_aug != src:
                            batch_output.append(f"{prefix_token} {src_aug} [TARGET] {tgt}\n")
                            lines_written += 1
                            
                except Exception:
                    continue
            
            # Tulis sekaligus per chunk agar I/O lebih ringan
            f_out.write("".join(batch_output))
            pbar.update(len(chunk))
            
            # Info Tambahan setiap 1 juta baris ke console (opsional)
            if lines_written % 1000000 < 50000:
                elapsed = time.time() - start_time
                print(f" | Info: {lines_written:,} baris tertulis | Waktu: {elapsed/60:.2f} menit")

        pbar.close()

    total_time = time.time() - start_time
    print(f"✅ Selesai {name}!")
    print(f"📊 Total Baris Tertulis: {lines_written:,}")
    print(f"⏱️ Total Waktu: {total_time/60:.2f} menit ({lines_written/total_time:.2f} baris/detik)")
# ==============================================================================
# 3. PROSES DATASET MONOLINGUAL (IndoNLP - Single Large File)
# ==============================================================================
def process_indo_monolingual():
    print(f"\n🇮🇩 Memproses File IndoNLP: {PATH_INDO_FILE.name}")
    output_file = OUTPUT_DIR / "processed_indo.txt"
    
    if not PATH_INDO_FILE.exists():
        print(f"❌ File tidak ditemukan: {PATH_INDO_FILE}")
        return

    lines_written = 0
    seen_hashes = set()

    with open(output_file, "w", encoding="utf-8") as f_out:
        # Baca file besar baris per baris (Stream)
        with open(PATH_INDO_FILE, "r", encoding="utf-8", errors="ignore") as f_in:
            for line in tqdm(f_in, desc="Membaca IndoNLG", unit=" lines"):
                cleaned = clean_text(line)
                if len(cleaned) < MIN_LEN: continue
                
                # Deduplikasi
                h = hash(cleaned)
                if h in seen_hashes: continue
                seen_hashes.add(h)
                if len(seen_hashes) > 10_000_000: seen_hashes.clear() 

                # Format [ID]
                f_out.write(f"[ID] {cleaned}\n")
                lines_written += 1

    print(f"✅ Selesai IndoNLP: {lines_written} baris tersimpan.")
# ==============================================================================
# EKSEKUSI UTAMA
# ==============================================================================
if __name__ == "__main__":
    # process_parallel_dataset("de_en", PATH_WMT_DE, "[EN2DE]", "de")
    # process_parallel_dataset("fr_en", PATH_WMT_FR, "[EN2FR]", "fr")
    process_indo_monolingual()
    
    #  Output Example
    # ID
    # [ID] Pemerintah mengumumkan kebijakan ekonomi baru tahun ini.
    
    # EN->FR
    # [EN2FR] The mahasiswa is membaca a buku [TARGET] L'étudiant lit un livre
    # [EN2FR] I ingin to pergi to the kota [TARGET] Je veux aller en ville
    
    # EN->DE
    # [EN2DE] The putih rumah is on the hill [TARGET] Das weiße Haus steht auf dem Hügel
    # [EN2DE] They cinta makan apples [TARGET] Sie lieben es, Äpfel zu essen
    
    print("\n🏁 PIPELINE SELESAI! Data siap di folder:", OUTPUT_DIR)