# Data Management Pipeline

Folder ini berisi semua aset terkait dataset, mulai dari data mentah hingga data yang siap untuk dilatih oleh model. Alur kerja data diatur secara sistematis melalui subfolder berikut.

---

## 📚 Sumber Dataset

Proyek ini menggunakan kombinasi dari dua sumber dataset utama untuk melatih dan mengevaluasi model bahasa:

### 1. WMT14 English-German (en-de)

- **Tujuan**: Digunakan sebagai dataset benchmark standar yang besar dan berkualitas tinggi. Meskipun ini adalah dataset terjemahan, kita akan menggunakan salah satu sisinya (misalnya, bahasa Inggris) sebagai corpus teks monolingual.
- **Sumber**: Anda dapat mengunduhnya melalui library `datasets` Hugging Face atau sumber resmi WMT.
- **Cara Mendapatkan (via `datasets`)**:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("wmt14", "de-en")
  ```
- **Penempatan**: Simpan file teks mentah (misalnya, `train.en`) di dalam folder `raw/`.

### 2. IndoNLP Corpora

- **Tujuan**: Untuk melatih model agar memiliki pemahaman yang kuat tentang Bahasa Indonesia. Kita akan menggunakan corpus teks yang tersedia dari koleksi IndoNLP.
- **Sumber**: Berbagai dataset teks dari repositori IndoNLP atau sumber lain seperti id-liputan6.
- **Cara Mendapatkan**: Unduh file teks atau CSV dari sumber yang dipilih. Jika dalam format CSV, ekstrak kolom teks yang relevan menjadi satu file `.txt`.
- **Penempatan**: Simpan file teks mentah (misalnya, `indonesian_corpus.txt`) di dalam folder `raw/`.

---

## 📂 Alur Kerja Folder

Data akan diproses melalui tahapan berikut, di mana setiap folder merepresentasikan satu keadaan data.

### `raw/`

**Isi**: File dataset asli yang diunduh langsung dari sumbernya, tanpa modifikasi apa pun.

- `wmt14_train.en`
- `indonesian_corpus.txt`

> **Catatan**: Folder ini diabaikan oleh `.gitignore` untuk mencegah file data besar masuk ke dalam repositori Git.

### `cleaned/`

**Isi**: Versi bersih dari dataset di `raw/` setelah diproses oleh `scripts/preprocess_data.py`.

- **Proses**: Cleaning (hapus noise), normalisasi (lowercase), dan deduplikasi.
- **Output**: Satu atau beberapa file teks besar (misalnya, `cleaned_corpus.txt`) yang berisi gabungan data bersih.

> Folder ini adalah input untuk melatih tokenizer (`tokenizer/train_tokenizer.py`).

### `tokenized/`

**Isi**: Dataset yang telah diubah menjadi urutan _token ID_ (integer) oleh tokenizer yang sudah dilatih.

- **Proses**: Teks dari `cleaned/` di-encode menggunakan `tokenizer/tokenizer.model`.
- **Output**: File dalam format biner atau memory-mapped (misalnya, `.bin` atau `.npy`) untuk efisiensi loading saat training.

> Ini adalah representasi data yang akan dibaca oleh `training/dataset.py`.

### `splits/`

**Isi**: File-file yang mendefinisikan pembagian dataset `tokenized/` menjadi set pelatihan (train), validasi (validation), dan pengujian (test).

- **Proses**: Membagi data tokenized menjadi beberapa bagian untuk memastikan evaluasi model yang objektif.
- **Output**: File-file seperti `train.bin`, `val.bin`.

> Folder ini adalah sumber data final yang akan di-load oleh PyTorch `DataLoader` dalam `training/train.py`.
