import torch
import numpy as np
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_path, seq_len):
        # Gunakan dtype=np.uint16 karena kita menyimpannya dengan tipe itu kemarin
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        
        # Validasi sederhana
        if len(self.data) <= seq_len:
            raise ValueError(f"Data di {data_path} (size: {len(self.data)}) terlalu kecil untuk seq_len {seq_len}")

    def __len__(self):
        # Pastikan hasil minimal adalah 0
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        # Ambil data dan konversi ke Long (int64) untuk PyTorch
        x = torch.from_numpy(self.data[idx : idx + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + self.seq_len + 1].astype(np.int64))
        return x, y
