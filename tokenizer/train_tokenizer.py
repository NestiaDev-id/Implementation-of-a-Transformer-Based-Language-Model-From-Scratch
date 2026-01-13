import sentencepiece as spm
from pathlib import Path

INPUT = "data/cleaned/sample_combined_for_tokenizer.txt"
MODEL_PREFIX = "tokenizer/multilingual"
VOCAB_SIZE = 50_000

# Pastikan folder output bersih
Path("tokenizer").mkdir(exist_ok=True)

spm.SentencePieceTrainer.train(
    input=INPUT,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=0.9995,
    split_digits=True,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    # user_defined_symbols=["<id>", "<en>", "<de>"],
)

print("✅ Tokenizer training untuk 3 bahasa selesai")
