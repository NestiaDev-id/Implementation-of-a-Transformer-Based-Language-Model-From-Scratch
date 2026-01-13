import sentencepiece as spm
from pathlib import Path

# Pastikan path model sudah benar
MODEL_PATH = "tokenizer/multilingual.model"

if not Path(MODEL_PATH).exists():
    print(f"❌ Error: Model file tidak ditemukan di {MODEL_PATH}")
    exit()

sp = spm.SentencePieceProcessor()
sp.load(MODEL_PATH)

sentences = [
    "hey gpt bisakah kamu mengajari bahasa german?",
    "can you teach me german?",
    "kannst du mir deutsch beibringen?",
    "bonjour peux-tu m'apprendre l'allemand?"
]

for s in sentences:
    print("-" * 40)
    print("TEXT:", s)
    
    # Gunakan encode_as_pieces untuk mendapatkan daftar token yang sebenarnya
    tokens = sp.encode_as_pieces(s)
    print("TOKENS:", tokens)
    
    if tokens:
        unk_count = tokens.count("<unk>")
        unk_ratio = unk_count / len(tokens)
        print("UNK RATIO: {:.2%}".format(unk_ratio))
    else:
        print("UNK RATIO: 0.00%")

